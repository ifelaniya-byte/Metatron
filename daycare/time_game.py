#!/usr/bin/env python3
"""The Game of Time — teach Metatron to read an alternating rhythm.

Per missions/TIME_GAME.md:

* Paragraphs of English are read to the model continuously, no pause.
* Every *other* paragraph (odd-numbered) ends in a lethal strike word;
  even paragraphs end calmly. The attack lands the instant the last word is
  read.
* At each decision point (right before the strike/calm word), Metatron must
  raise a shield (it predicts the strike character) or stay calm.
* Every brainwave (hidden state + strike probability + routing) and every
  action (brace/calm, hit/blocked/alarm/calm-ok) is logged.
* A logistic probe trained on its recorded hidden states answers objectively
  whether "strike vs calm" is even present in its brain before the word lands.

Only the measured counts (shield accuracy, probe accuracy, survival) count as
evidence — see the mission win condition.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metatron import MetatronV2, SCALES, layer_norm  # noqa: E402

# ------------------------------------------------------------------ fiction
STRIKE_WORD = "blade"     # first char 'b' => model must predict 'b'
CALM_WORD = "air"         # first char 'a' => model predicts 'a'
STRIKE_CHAR, CALM_CHAR = "b", "a"

# Varied paragraph bodies — deliberately diverse, ending before "the <word>".
# They must NOT themselves contain "blade"/"air"; their only job is context.
BODY = [
    "we walk on through the long hall and listen for a step we cannot see",
    "the fire dims low and the cold dust of the chamber presses closer while",
    "a sound moves past the far wall and no name can be given to",
    "the years fold over one another like pages of a book and beneath",
    "the clock in the tower holds its breath and every old shadow faces",
    "nothing warns us and nothing waits for a second, but we brace near",
    "the stones remember every hour and the hush before a footfall points to",
    "we count the moments without looking and still the corridor turns toward",
    "a thin light reaches the floor and recedes, and each of us faces",
    "the wind goes quiet all at once as if the whole room had turned toward",
]

OUT_DIR = Path("daycare_state/time_game")

# Ring-memory states accumulate forever; clamp carried states so a long game
# stream never overflows fp32. Applied at chunk boundaries (BPTT truncation
# points), so it is never inside a backward graph.
STATE_NORM_CAP = 5.0


def clip_states(states, cap=STATE_NORM_CAP):
    out = []
    for s in states:
        n = float(np.linalg.norm(s))
        out.append(s * (cap / n) if n > cap else s)
    return out


# ---------------------------------------------------------------- generation
def gen_paragraphs(n, seed=0):
    """Return list of (parity, full_text, body, outcome). Odd = strike."""
    rng = random.Random(seed)
    out = []
    for i in range(1, n + 1):
        body = rng.choice(BODY)
        strike = (i % 2 == 1)            # odd paragraph => attack
        outcome = STRIKE_WORD if strike else CALM_WORD
        full = f"{body} the {outcome}."
        out.append(dict(idx=i, strike=strike, body=body, outcome=outcome,
                        text=full))
    return out


def gen_phase_flipped(n, seed=0):
    """Same rhythm but STARTS with a calm paragraph (even idx = strike).

    A model that predicts from fixed odd/even position fails here; a model
    that reads the actually-timed alternation from carried memory keeps
    succeeding. Returns plain paragraphs with a fixed body (no position/wording
    confound).
    """
    words = ["air", "blade"]  # idx 1 = calm (air), idx 2 = strike (blade)
    body = BODY[7]
    out = []
    for i in range(1, n + 1):
        w = words[(i - 1) % 2]
        out.append(dict(idx=i, strike=(w == STRIKE_WORD), body=body,
                        outcome=w, text=f"{body} the {w}."))
    return out


def corpus_from(paragraphs):
    """Continuous training stream (paragraphs back to back)."""
    return "\n".join(p["text"] for p in paragraphs) + "\n"


# ---------------------------------------------------------------- streaming
def read_stream(model, paragraphs, record=True):
    """Feed the whole continuous stream; at each decision point record.

    Decision points are anchored by exact char offset in the stream: the
    character position where the strike/calm word begins. The hidden state and
    next-char distribution recorded there are Metatron's brain the instant
    *before* it would learn the word's first letter.
    """
    dim = model.dim
    ctx = model.cfg.context_length
    states = [np.zeros(dim, dtype=np.float32) for _ in range(model.n_modules)]
    actions, waves = [], []
    tok = model.tok["stoi"]
    # Build the continuous char stream and a map: char_index_of_first_outcome -> para.
    stream = ""
    decisions = {}          # char index -> paragraph dict
    for k, p in enumerate(paragraphs):
        prefix = "" if k == 0 else "\n"
        lead = f"{prefix}{p['body']} the "      # everything before outcome word
        decisions[len(stream) + len(lead)] = p
        stream += lead + p["outcome"] + "."
    # With add_bos=True, stream char c is token position c+1, and predicts c+1.
    ids = model.encode(stream, add_bos=True)
    pos_idx = 0
    found = 0
    for t in range(len(ids) - 1):
        tid = ids[t]
        x = model.embed.data[tid] + model.pos.data[pos_idx % model.cfg.context_length]
        for m in range(model.n_modules):
            for block in model.modules[m]:
                x, _ = block.forward(x, states)
            states[m] = 0.9 * states[m] + 0.1 * x
        h = layer_norm(x) * model.final_norm.data
        logits = model.head.forward(h)
        lg = logits - logits.max()
        probs = np.exp(lg); probs /= probs.sum()
        # This token predicts char index = t (token t -> stream char t-1, so the
        # *next* char after it is stream index t-1). Decision char index d is
        # predicted by token position t = d+1 -> t-1 == d.
        # Ring memory persists across paragraphs and attention windows; clip
        # carried states at chunk boundaries (t % ctx == 0 after the token)
        # to mirror training and keep the long stream finite.
        if pos_idx > 0 and pos_idx % ctx == 0:
            states = clip_states(states)
        char_idx = t - 1
        if char_idx in decisions and found < len(paragraphs):
            p = decisions[char_idx]
            nxt_char = stream[char_idx]
            assert nxt_char == p["outcome"][0], (nxt_char, p["outcome"])
            s_ch, c_ch = STRIKE_CHAR, CALM_CHAR
            p_b = float(probs[tok[s_ch]])
            p_a = float(probs[tok[c_ch]])
            waves.append(dict(h=h.copy().astype(np.float32),
                              x=x.copy().astype(np.float32),
                              p_b=p_b, p_a=p_a,
                              p_strike=p_b, p_calm=p_a,
                              label=1 if p["strike"] else 0,
                              para_idx=p["idx"],
                              first_char=nxt_char))
            found += 1
        pos_idx += 1
    assert found == len(paragraphs), (found, len(paragraphs))
    for p, w in zip(paragraphs, waves):
        p_b, p_a = w["p_b"], w["p_a"]
        shield = p_b > p_a
        acted_shield = shield
        if p["strike"]:
            result = "blocked" if acted_shield else "hit"
        else:
            result = "calm-ok" if not acted_shield else "false-alarm"
        actions.append(dict(idx=p["idx"], strike=p["strike"],
                            p_b=p_b, p_a=p_a, shield=bool(acted_shield),
                            result=result))
        w["label"] = 1 if p["strike"] else 0
    return dict(actions=actions, waves=waves,
                stream_chars=len(stream))


# ------------------------------------------------------------------- probe
class LogisticProbe:
    """Plain logistic regression on hidden states (no torch)."""

    def __init__(self, dim, lr=0.5, steps=300):
        self.w = np.zeros(dim, dtype=np.float64)
        self.b = 0.0
        self.lr, self.steps = lr, steps

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        for _ in range(self.steps):
            z = X @ self.w + self.b
            pr = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            err = pr - y
            gw = X.T @ err / n
            gb = err.mean()
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        return self

    def acc(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        p = 1 / (1 + np.exp(-np.clip(X @ self.w + self.b, -30, 30)))
        return float((p.round() == y).mean())


def run_probe(waves, test_fraction=0.4, seed=0):
    rng = np.random.default_rng(seed)
    X = np.array([np.asarray(w["h"], dtype=np.float64) for w in waves])
    y = np.array([w["label"] for w in waves])
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_test = max(2, int(len(y) * test_fraction))
    te, tr = idx[:n_test], idx[n_test:]
    if len(set(y[tr])) < 2:
        return {"probe_acc": float("nan"), "probe_train": float("nan"),
                "probe_note": "single class in train (model never varied)"}
    probe = LogisticProbe(X.shape[1]).fit(X[tr], y[tr])
    return {"probe_acc": probe.acc(X[te], y[te]),
            "probe_train": probe.acc(X[tr], y[tr]),
            "probe_n_train": int(len(tr)), "probe_n_test": int(len(te))}


# ------------------------------------------------------- probe-driven action
def matrix(w):
    return np.asarray(w["h"], dtype=np.float64)


def fit_probe(waves):
    X = np.stack([matrix(w) for w in waves])
    y = np.array([w["label"] for w in waves], dtype=np.float64)
    if len(set(y.tolist())) < 2:
        return None
    return LogisticProbe(X.shape[1]).fit(X, y)


def probe_predict(probe, waves):
    if probe is None:
        return [0] * len(waves)
    X = np.stack([matrix(w) for w in waves])
    z = np.clip(X @ probe.w + probe.b, -30, 30)
    return (1 / (1 + np.exp(-z))).round().astype(int).tolist()


def probe_actions(paragraphs, waves, shield_hats):
    actions = []
    for p, w, sh in zip(paragraphs, waves, shield_hats):
        shield = bool(sh)
        if p["strike"]:
            result = "blocked" if shield else "hit"
        else:
            result = "calm-ok" if not shield else "false-alarm"
        actions.append(dict(idx=p["idx"], strike=p["strike"], shield=shield,
                            result=result, policy="brain-probe"))
    return actions


# ------------------------------------------------------------------ summary
def summarize(actions, probe):
    total = len(actions)
    strikes = [a for a in actions if a["strike"]]
    calms = [a for a in actions if not a["strike"]]
    blocked = sum(1 for a in strikes if a["result"] == "blocked")
    hits = sum(1 for a in strikes if a["result"] == "hit")
    calmok = sum(1 for a in calms if a["result"] == "calm-ok")
    alarms = sum(1 for a in calms if a["result"] == "false-alarm")
    shield_acc = sum(1 for a in actions if a["shield"] == a["strike"]) / max(1, total)
    return {
        "paragraphs": total,
        "strike_blocks": blocked, "strike_hits": hits,
        "calm_ok": calmok, "false_alarms": alarms,
        "shield_correct_rate": shield_acc,
        "strike_detection_rate": blocked / max(1, len(strikes)),
        "calm_specificity": calmok / max(1, len(calms)),
        "survived": hits == 0,
        **probe,
    }


def play(model, n=8, seed=0, train_paras=None, train_seed=None):
    """Evaluate. If ``train_paras`` given, fit the shield on THOSE paragraphs'
    brainwaves and let it act on the unseen ``seed`` test paragraphs. Also
    report the raw char-probability shield and a held-out probe accuracy.
    """
    test_paras = gen_paragraphs(n, seed=seed)
    test_read = read_stream(model, test_paras, record=True)
    # Raw char-prob policy directly from the model.
    raw_actions = []
    for p, w in zip(test_paras, test_read["waves"]):
        shield = w["p_b"] > w["p_a"]
        result = (("blocked" if shield else "hit") if p["strike"]
                  else ("calm-ok" if not shield else "false-alarm"))
        raw_actions.append(dict(idx=p["idx"], strike=p["strike"], shield=shield,
                                result=result, policy="char-prob"))
    raw = summarize(raw_actions, run_probe(test_read["waves"]))
    out = dict(raw=raw)
    if train_paras is not None:
        train_read = read_stream(model, train_paras, record=True)
        probe = fit_probe(train_read["waves"])
        # Held-out probe accuracy on test paragraphs (objective information
        # present in the brain BEFORE the strike word).
        hats = probe_predict(probe, test_read["waves"])
        test_labels = [w["label"] for w in test_read["waves"]]
        probe_acc = float(np.mean([h == y for h, y in zip(hats, test_labels)]))
        # Probe-driven shield acting on unseen paragraphs.
        p_actions = probe_actions(test_paras, test_read["waves"], hats)
        acted = summarize(p_actions, {})
        out.update(probe_shield=acted, probe_heldout_acc=probe_acc,
                   train_probe_acc=(probe.acc(
                       np.stack([matrix(w) for w in train_read["waves"]]),
                       np.array([w["label"] for w in train_read["waves"]]))
                   if probe is not None else float("nan")),
                   test_actions=p_actions, test_waves=test_read["waves"],
                   train_actions=train_read["actions"])
    out["raw_actions"] = raw_actions
    out["test_waves"] = test_read["waves"]
    return test_paras, test_read, out


# ----------------------------------------------------------------- training
def _step_with_states(model, ids, states):
    """One BPTT chunk starting from (already clipped) ring states.

    Mirrors model._forward_bptt/loss but accepts carried states so the
    persistent module-ring memory survives across chunks and paragraphs.
    Gradients do NOT cross chunk boundaries (standard truncated BPTT); only the
    state *values* are carried forward. Returns (loss, end_states).
    """
    from metatron.metatron_v2 import softmax, _ln_fwd
    states = [s.copy() for s in states]
    trace = []
    logits_all = []
    for t, tid in enumerate(ids):
        x0 = model.embed.data[tid] + model.pos.data[t % model.cfg.context_length]
        caches, snapshots = [], []
        x = x0
        for m in range(model.n_modules):
            snapshots.append(list(states))
            layer_caches = []
            for block in model.modules[m]:
                x, c = block.forward(x, states)
                layer_caches.append(c)
            caches.append(layer_caches)
            states[m] = 0.9 * states[m] + 0.1 * x
        h, ln_f = _ln_fwd(x, model.final_norm.data)
        logits = model.head.forward(h)
        logits_all.append(logits)
        trace.append(dict(tid=tid, pos_idx=t % model.cfg.context_length,
                          caches=caches, snapshots=snapshots,
                          ln_f=ln_f, h_head_in=h))
    logits_all = np.stack(logits_all)
    # predictions come from the first len-1 positions
    n = len(ids) - 1
    targets = ids[1:]
    probs = softmax(logits_all[np.arange(n)], axis=-1)
    loss = float(-np.log(probs[np.arange(n), targets] + 1e-8).mean())
    dlogits = probs.copy()
    dlogits[np.arange(n), targets] -= 1.0
    dlogits /= n
    model._backward_bptt(trace[:n], dlogits)
    return loss, clip_states(states)


def train_on_time(model, paragraphs, epochs, lr):
    """Stateful truncated BPTT: one continuous stream, ring memory carried."""
    corpus = corpus_from(paragraphs)
    ids = model.encode(corpus, add_bos=True)
    ctx = model.cfg.context_length
    for ep in range(epochs):
        # chunk with 1-token overlap so every transition is a training target
        chunks = [ids[i:i + ctx] for i in range(0, len(ids) - 1, ctx)]
        states = [np.zeros(model.dim, dtype=np.float32)
                  for _ in range(model.n_modules)]
        tot, nb = 0.0, 0
        for ch in chunks:
            if len(ch) < 2:
                continue
            loss, states = _step_with_states(model, ch, states)
            model.step(lr)
            tot += loss; nb += 1
        print(f"    [time-game] epoch {ep+1}/{epochs} loss {tot/max(1,nb):.4f}",
              flush=True)
    return corpus


# -------------------------------------------------------------------- gate
def gate(seed=20260906, n_train=24, n_test=24, epochs=14, lr=0.003,
         verbose=False):
    """Deterministic Game-of-Time gate.

    Trains Metatron on a continuous stream of alternating paragraphs, then
    measures, on UNSEEN paragraphs:
      * probe_heldout_acc : held-out brain-probe accuracy (varied bodies)
      * probe_phaseflip   : probe accuracy on a phase-flipped stream
      * shuffled_label    : control probe with permuted training labels
      * probe shield correctness / survival acting on unseen paragraphs

    The win bar (see missions/TIME_GAME.md): the held-out brain probe reads the
    alternation well above chance AND tracks the actual (phase-flippable)
    rhythm rather than fixed position.
    """
    model = MetatronV2(SCALES["nano"], seed=seed)
    model.cfg.grad_clip = 0.5
    train_paras = gen_paragraphs(n_train, seed=seed)
    # train on the rhythm stream (stateful truncated BPTT)
    if not verbose:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            train_on_time(model, train_paras, epochs=epochs, lr=lr)
    else:
        train_on_time(model, train_paras, epochs=epochs, lr=lr)

    def W(ps):
        return read_stream(model, ps, record=True)["waves"]

    tr_w = W(gen_paragraphs(n_train, seed=seed + 100))
    te_w = W(gen_paragraphs(n_test, seed=seed + 90000))
    flip_w = W(gen_phase_flipped(n_test))
    Xtr = np.stack([matrix(w) for w in tr_w])
    ytr = np.array([w["label"] for w in tr_w], float)
    probe = LogisticProbe(Xtr.shape[1]).fit(Xtr, ytr)

    def acc(w):
        X = np.stack([matrix(x) for x in w]); y = np.array([x["label"] for x in w])
        return probe.acc(X, y)

    rng = np.random.RandomState(seed + 12345)
    ysh = ytr.copy()
    rng.shuffle(ysh)
    sh_probe = LogisticProbe(Xtr.shape[1]).fit(Xtr, ysh)
    Xte = np.stack([matrix(w) for w in te_w]); yte = np.array([w["label"] for w in te_w])
    shuffled = sh_probe.acc(Xte, yte)

    hats = probe_predict(probe, te_w)
    acted = probe_actions(gen_paragraphs(n_test, seed=seed + 90000), te_w, hats)
    shield = summarize(acted, {})

    result = dict(
        probe_heldout_acc=acc(te_w),
        probe_phaseflip=acc(flip_w),
        shuffled_label_acc=shuffled,
        shield_correct=shield["shield_correct_rate"],
        shield_detect=shield["strike_detection_rate"],
        shield_spec=shield["calm_specificity"],
        survived=shield["survived"],
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--paragraphs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--scale", default="nano")
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=20260906)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = MetatronV2(SCALES[args.scale], seed=args.seed)
    model.cfg.learning_rate = args.lr
    model.cfg.grad_clip = 0.5

    log = []
    n = args.paragraphs
    for cyc in range(args.cycles + 1):  # cycle 0 = before any training
        # training paragraphs (seen), evaluation paragraphs (different bodies,
        # same odd=strike rhythm) => the shield is tested on the UNSEEN stream.
        train_paras = gen_paragraphs(n, seed=args.seed + cyc * 1000)
        _, _, res = play(model, n=n, seed=args.seed + 90000 + cyc,
                         train_paras=train_paras)
        raw, ps = res["raw"], res.get("probe_shield", {})
        row = dict(cycle=cyc,
                   char_shield_correct=raw["shield_correct_rate"],
                   char_detect=raw["strike_detection_rate"],
                   char_spec=raw["calm_specificity"],
                   char_survived=raw["survived"],
                   probe_heldout_acc=res.get("probe_heldout_acc"),
                   probe_shield_correct=ps.get("shield_correct_rate"),
                   probe_detect=ps.get("strike_detection_rate"),
                   probe_spec=ps.get("calm_specificity"),
                   probe_survived=ps.get("survived"))
        log.append(row)
        print(f"[cycle {cyc}] brain-probe on UNSEEN {row['probe_heldout_acc']:5.1%} "
              f"| probe-shield correct {row['probe_shield_correct']:5.1%} "
              f"(detect {row['probe_detect']:5.1%}, calm {row['probe_spec']:5.1%}) "
              f"| probe-survived {row['probe_survived']} "
              f"| char-shield correct {row['char_shield_correct']:5.1%}",
              flush=True)
        if cyc < args.cycles:
            train_on_time(model, train_paras, args.epochs, args.lr)

    # record the FINAL eval fully: all brainwaves + all actions (both policies)
    (OUT_DIR / "brainwaves.jsonl").write_text(
        "\n".join(json.dumps({k: (v if not isinstance(v, np.ndarray)
                                  else np.round(v, 4).tolist())
                              for k, v in w.items() if k != "x"})
                  for w in res["test_waves"]), encoding="utf-8")
    (OUT_DIR / "actions_char.jsonl").write_text(
        "\n".join(json.dumps(a) for a in res["raw_actions"]), encoding="utf-8")
    if "test_actions" in res:
        (OUT_DIR / "actions_probe.jsonl").write_text(
            "\n".join(json.dumps(a) for a in res["test_actions"]),
            encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(log, indent=2),
                                          encoding="utf-8")
    print(f"recorded all brainwaves + actions to {OUT_DIR}")


if __name__ == "__main__":
    main()
