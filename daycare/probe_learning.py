#!/usr/bin/env python3
"""Probe what the Metatron champion actually learned.

Reports evidence along three axes:
  1. distribution  — does it expect the letter distribution / n-grams of English?
  2. memorization  — held-out split of its curriculum vs repeated lines;
  3. abstraction   — novel English text vs scrambled/reversed/random controls.

All numbers are mean cross-entropy loss in nats over bounded-context windows
(identical geometry to the training gate), plus argmax next-char accuracy.
"""
from __future__ import annotations
import math, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metatron import MetatronV2, SCALES, make_batches, DEFAULT_TEXT  # noqa

CHAMP = Path("daycare_state/champion/model.pkl")

# ---- corpora -------------------------------------------------------------
lines = [l.strip() for l in DEFAULT_TEXT.splitlines() if l.strip()]
train_split = "\n".join(lines[:-5])
seen_split = "\n".join(lines[-5:])  # part of DEFAULT_TEXT => seen during training

NOVEL_EN = ("""
the quiet river carved a narrow valley through old stone and pine forest
every morning the baker warmed the oven before the village had fully woken
a student learns mathematics by working problems and checking each answer
the harbor filled with fishing boats as the tide rose under grey autumn skies
""").strip()

NOVEL_OTHER = ("""
the cellular tower relays packets between handsets across the microwave band
quantum entanglement links paired particles regardless of distance between them
""").strip()

REVERSED = NOVEL_EN[::-1]
WORDSCRAMBLED = "\n".join("".join(np.random.default_rng(i).permutation(list(w)))
                          for i, w in enumerate(NOVEL_EN.split("\n")))
LETTERS = "abcdefghijklmnopqrstuvwxyz \n"
RANDOM = "\n".join("".join(np.random.default_rng(i).choice(list(LETTERS), size=55))
                   for i in range(4))

THEORETICAL_UNIFORM = math.log(256)  # vocab_size 256, no conditioning


def windows(model, text):
    return [seq for b in make_batches(text, model, model.cfg.context_length,
                                      model.cfg.batch_size) for seq in b]


def eval_loss(model, text, also_acc=False):
    loss_sum, n, correct, total = 0.0, 0, 0, 0
    for seq in windows(model, text):
        inp, tgt = seq[:-1], seq[1:]
        logits = model.forward(inp)
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        loss_sum += -np.log(probs[np.arange(len(tgt)), tgt] + 1e-8).sum()
        n += len(tgt)
        if also_acc:
            correct += (probs.argmax(axis=-1) == np.array(tgt)).sum()
            total += len(tgt)
    loss = loss_sum / max(1, n)
    acc = correct / max(1, total) if also_acc else None
    return float(loss), acc


def main():
    champ = pickle.load(open(CHAMP, "rb"))
    fresh = MetatronV2(SCALES["nano"], seed=12345)

    print("=== 1. GLOBAL SURPRISAL BY CORPUS (loss in nats, lower = expected) ===")
    print(f"{'corpus':28s} {'champion':>9s} {'untrained':>10s} {'bits/ch':>8s}")
    for name, txt in [
        ("train curriculum (seen)", train_split),
        ("other curriculum lines (seen)", seen_split),
        ("novel English", NOVEL_EN),
        ("novel technical English", NOVEL_OTHER),
        ("word-scrambled English", WORDSCRAMBLED),
        ("reversed English", REVERSED),
        ("uniform random letters", RANDOM),
    ]:
        lc, acc = eval_loss(champ, txt, also_acc=True)
        lf, _ = eval_loss(fresh, txt)
        print(f"{name:28s} {lc:9.3f} {lf:10.3f} {lc/math.log(2):8.3f}"
              f"   top1-acc={acc:.1%}")
    print(f"\ntheoretical uniform over 256 vocab = {THEORETICAL_UNIFORM:.2f} nats"
          f" = {THEORETICAL_UNIFORM/math.log(2):.2f} bits/char")

    print("\n=== 2. PER-SENTENCE MEMORIZATION (champion loss per line) ===")
    rows = []
    for line in lines:
        l, acc = eval_loss(champ, line, also_acc=True)
        rows.append((l, acc, line))
    rows.sort()
    print("best learned (lowest surprise):")
    for l, acc, line in rows[:4]:
        print(f"  {l:.3f} nats acc {acc:4.0%}  {line[:62]}")
    print("highest surprise:")
    for l, acc, line in rows[-3:]:
        print(f"  {l:.3f} nats acc {acc:4.0%}  {line[:62]}")

    print("\n=== 3. CONTEXT SENSITIVITY (prefix -> champion surprisal) ===")
    probes = [
        ("the flower of life", " the"),
        ("the flower of life", " zxq"),
        ("neural networks", " learn"),
        ("neural networks", " qkxw"),
        ("sacred geometry", " is"),
        ("sacred geometry", " bz"),
    ]
    for prefix, cont in probes:
        ids = champ.encode(prefix + cont, add_bos=True)
        k = len(champ.encode(prefix, add_bos=True))
        logits = champ.forward(ids[:-1])
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        surprisal = -np.log(probs[np.arange(len(ids)-1), ids[1:]] + 1e-8)[k-1:].mean()
        topk = np.argsort(-probs[k-1])[:5]
        guess = "".join(champ.tok["itos"][i] for i in topk
                        if not champ.tok["itos"][i].startswith("<"))[:5]
        print(f"  prefix {prefix!r:26s} cont {cont!r:6s} surprisal "
              f"{surprisal:5.2f} nats  top-guesses={guess!r}")

    print("\n=== 4. CHAR-UNIGRAM SKEW: trained weights expect English letters ===")
    # Mean predicted distribution over the whole curriculum, aggregated
    agg = np.zeros(champ.cfg.vocab_size)
    cnt = 0
    for seq in windows(champ, train_split):
        logits = champ.forward(seq[:-1])
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        agg += p.sum(axis=0); cnt += p.shape[0]
    agg /= cnt
    letters_idx = [champ.tok["stoi"][c] for c in "etaoin shrdlu"]
    weird = [champ.tok["stoi"].get(c, -1) for c in "qjzx%&@"]
    print("  mean prob on English-frequent chars (etaoin space shrdlu):",
          f"{sum(agg[i] for i in letters_idx):.3f}")
    print("  mean prob on rare/symbol chars (q j z x % & @):",
          f"{sum(agg[i] for i in weird if i>=0):.5f}")


if __name__ == "__main__":
    main()
