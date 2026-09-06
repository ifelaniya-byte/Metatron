#!/usr/bin/env python3
"""Speech recitation + quiz literacy (fully objective gates).

Metatron studies tough speeches stored as Markdown under ``speeches/``. Each
file carries the public-domain text and a quiz whose answers are exact words
from that text. Two gates score it, both deterministic:

  * **recitation** — given the start of a sentence, greedy-continue it and
    measure verbatim overlap with the true continuation (char accuracy and
    exact-word rate).
  * **quiz** — given a cloze question ("...brought forth a new ____"), emit
    the missing word; scored by exact match against the answer key.

Training corpus (``--build-corpus``) repeats the speech lines and the
question/answer pairs densely, which is what a small char model needs to
memorize and reproduce. Nothing here is judged subjectively — recitation is
verbatim or not, and an answer matches the key or it does not.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEECH_DIR = ROOT / "speeches"

_WORD_RE = re.compile(r"[a-z']+")
RECITE_TARGET = 0.90   # verbatim char accuracy to call a line "recited"
QUIZ_TARGET = 0.95     # quiz answer accuracy to call a speech "known"


# ------------------------------------------------------------------- parse
def parse_speech(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("# ").strip()
    # Speech body sits between the "## Text" heading and "## Quiz".
    body = text.split("## Text", 1)[1] if "## Text" in text else ""
    body = body.split("## Quiz", 1)[0]
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    # Quiz pairs.
    quiz: list[tuple[str, str]] = []
    q = None
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("Q:"):
            q = ln[2:].strip()
        elif ln.startswith("A:") and q is not None:
            quiz.append((q, ln[2:].strip().lower()))
            q = None
    return {"file": path.name, "title": title, "lines": lines, "quiz": quiz}


def all_speeches():
    # Only Markdown files that actually carry a Text + Quiz block (this
    # excludes index files such as speeches/README.md).
    out = []
    for p in sorted(SPEECH_DIR.glob("*.md")):
        sp = parse_speech(p)
        if sp["lines"] and sp["quiz"]:
            out.append(sp)
    return out


# ----------------------------------------------------------------- corpus
def _norm(s: str) -> str:
    return s.replace("______", "").strip()


def build_quiz_pair(question: str, answer: str) -> str:
    # Cloze sentence -> "answer <word>". The blank marker is dropped so the
    # model sees a clean question then the answer word.
    q = _norm(question)
    return f"Q: {q}\nA: {answer}"


def build_corpus(speeches=None, repeats=6, quiz_repeats=10):
    speeches = speeches or all_speeches()
    blocks = []
    for _ in range(repeats):
        rec = []
        for sp in speeches:
            for ln in sp["lines"]:
                rec.append(ln)
        blocks.append("\n".join(rec))
    for _ in range(quiz_repeats):
        qa = []
        for sp in speeches:
            for q, a in sp["quiz"]:
                qa.append(build_quiz_pair(q, a))
        blocks.append("\n".join(qa))
    return "\n".join(blocks) + "\n"


# ----------------------------------------------------------------- gates
def _greedy(model, prompt, max_new):
    import numpy as np
    saved = getattr(model, "_rng", None)
    model._rng = np.random.default_rng(0)
    try:
        return model.generate(prompt, max_new=max_new, temperature=1e-3, top_k=1)
    finally:
        model._rng = saved


def evaluate_recitation(model, speeches=None):
    """Line-by-line: prompt with the first ~40% of words, score the rest."""
    speeches = speeches or all_speeches()
    char_ok = char_tot = word_ok = word_tot = 0
    exact_lines = 0
    n_lines = 0
    per_speech = []
    for sp in speeches:
        sp_char = sp_n = sp_exact = 0
        for ln in sp["lines"]:
            words = ln.split()
            if len(words) < 6:
                continue
            cut = max(2, len(words) // 3)
            prompt = " ".join(words[:cut])
            target = " " + " ".join(words[cut:])
            out = _greedy(model, prompt, max_new=len(target) + 4)
            cont = out[len(prompt):].split("\n")[0]
            L = len(target)
            c = sum(1 for a, b in zip(cont[:L], target) if a == b)
            char_ok += c; char_tot += L; sp_char += c; sp_n += L
            tw = _WORD_RE.findall(target.lower())
            cw = _WORD_RE.findall(cont[:L].lower())
            for i, w in enumerate(tw):
                word_tot += 1
                if i < len(cw) and cw[i] == w:
                    word_ok += 1
            n_lines += 1
            if cont.strip().startswith(target.strip()[:min(len(target.strip()), 12)]):
                exact_lines += 1; sp_exact += 1
        if sp_n:
            per_speech.append((sp["title"], sp_char / sp_n, sp_exact))
    return {
        "recite_char_acc": char_ok / max(1, char_tot),
        "recite_word_acc": word_ok / max(1, word_tot),
        "recite_lines_started_exact": exact_lines / max(1, n_lines),
        "recite_lines_scored": float(n_lines),
        "per_speech": per_speech,
    }


def evaluate_quiz(model, speeches=None):
    """Cloze -> answer word, exact match against the key."""
    speeches = speeches or all_speeches()
    ok = tot = 0
    misses = []
    for sp in speeches:
        for q, a in sp["quiz"]:
            prompt = "Q: " + _norm(q) + "\nA:"
            out = _greedy(model, prompt, max_new=8)
            ans = out[len(prompt):].strip().split("\n")[0]
            words = _WORD_RE.findall(ans.lower())
            pred = words[0] if words else ""
            good = (pred == a)
            ok += good; tot += 1
            if not good:
                misses.append((sp["title"][:24], q[:40], pred or "—", a))
    return {
        "quiz_acc": ok / max(1, tot),
        "quiz_scored": float(tot),
        "quiz_misses": misses,
    }


def evaluate_speeches(model, speeches=None):
    r = evaluate_recitation(model, speeches)
    q = evaluate_quiz(model, speeches)
    r.update(q)
    r["speech_literate"] = 1.0 if (r["quiz_acc"] >= QUIZ_TARGET
                                   and r["recite_char_acc"] >= RECITE_TARGET) else 0.0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-corpus", default=None,
                    help="write a training corpus to this path")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--quiz-repeats", type=int, default=10)
    ap.add_argument("--checkpoint", default=None,
                    help="model.pkl to evaluate")
    args = ap.parse_args()

    speeches = all_speeches()
    print(f"loaded {len(speeches)} speeches:")
    for sp in speeches:
        print(f"  - {sp['title']}: {len(sp['lines'])} lines, {len(sp['quiz'])} quiz")

    if args.build_corpus:
        corpus = build_corpus(speeches, args.repeats, args.quiz_repeats)
        p = Path(args.build_corpus)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(corpus, encoding="utf-8")
        print(f"wrote {p} ({len(corpus)} chars)")

    if args.checkpoint:
        import pickle
        with open(args.checkpoint, "rb") as f:
            model = pickle.load(f)
        import json
        r = evaluate_speeches(model, speeches)
        print(json.dumps({k: v for k, v in r.items() if k != "per_speech"},
                         indent=2))
        for title, acc, exact in r["per_speech"]:
            print(f"  recite {title[:40]:40s} char {acc:5.1%}")
        if r["quiz_misses"]:
            print("  quiz misses (title, question, pred, key):")
            for m in r["quiz_misses"][:8]:
                print("   ", m)


if __name__ == "__main__":
    main()
