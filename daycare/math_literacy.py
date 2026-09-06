#!/usr/bin/env python3
"""Math literacy: teach Metatron arithmetic in digits AND English words.

This is the verifiable route to language. Unlike free prose, every math
statement can be checked by a calculator, so the gate is fully objective:

    2 + 3 = 5                 (symbolic)
    two plus three equals five (English)

A generation is correct only if its parsed answer equals the real answer.
No one counts words or guesses intent — the answer is right or it is wrong.
That makes every English math sentence the model emits grounded in a
verifiable fact.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen"]
WORD_TO_NUM = {w: i for i, w in enumerate(WORDS)}
NUM_TO_WORD = {i: w for i, w in enumerate(WORDS)}
MAX_RESULT = len(WORDS) - 1  # 18


def to_word(n: int) -> str:
    return NUM_TO_WORD[n]


def _all_problems(max_operand=9):
    probs = []
    for a in range(max_operand + 1):
        for b in range(max_operand + 1):
            if a + b <= MAX_RESULT:
                probs.append((a, "plus", b, a + b))
            if a >= b:
                probs.append((a, "minus", b, a - b))
    return probs


def split_problems(max_operand=9, hold_out=0.2, seed=20260906):
    probs = _all_problems(max_operand)
    rng = random.Random(seed)
    rng.shuffle(probs)
    n_hold = int(len(probs) * hold_out)
    # Hold out a fixed, stratified slice (both ops).
    test = probs[:n_hold]
    train = probs[n_hold:]
    return train, test


def symbolic(p):
    a, op, b, r = p
    sym = "+" if op == "plus" else "-"
    return f"{a} {sym} {b} = {r}"


def english(p):
    a, op, b, r = p
    return f"{to_word(a)} {op} {to_word(b)} equals {to_word(r)}"


def build_corpus(problems, repeats=6, seed=7, english_ratio=0.5,
                 rng_seed_lines=False):
    rng = random.Random(seed)
    lines = []
    for _ in range(repeats):
        block = []
        for p in problems:
            use_en = rng.random() < english_ratio
            block.append(english(p) if use_en else symbolic(p))
        rng.shuffle(block)
        lines.append("\n".join(block))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ parsing
_TRAIL = re.compile(r"[a-z0-9 ,.]+$")


def parse_answer(text: str, prompt_len: int):
    """Extract the model's answer (int) from a completion, or None."""
    cont = text[prompt_len:]
    # Stop at a newline or extra padding.
    cont = cont.split("\n")[0]
    # Prefer a trailing digit token (symbolic form).
    digits = re.findall(r"\d+", cont)
    if digits:
        return int(digits[-1])
    # Otherwise the last English number word.
    words = re.findall(r"[a-z]+", cont)
    for w in reversed(words):
        if w in WORD_TO_NUM:
            return WORD_TO_NUM[w]
    return None


def generate_answer(model, prompt, max_new=24):
    m_rng = getattr(model, "_rng", None)
    saved = m_rng
    import numpy as np
    model._rng = np.random.default_rng(0)
    try:
        out = model.generate(prompt, max_new=max_new, temperature=1e-3, top_k=1)
    finally:
        model._rng = saved
    return out


def evaluate_math(model, problems, english_ratio=0.5, seed=11):
    """Greedy-answer every problem (mixing formats deterministically).

    Returns dict with overall / symbolic / english exact-answer accuracy.
    """
    rng = random.Random(seed)
    total = sym_ok = en_ok = 0
    sym_tot = en_tot = 0
    for p in problems:
        a, op, b, r = p
        use_en = rng.random() < english_ratio
        if use_en:
            prompt = f"{to_word(a)} {op} {to_word(b)} equals"
            sym_tot_local = False
        else:
            sym = "+" if op == "plus" else "-"
            prompt = f"{a} {sym} {b} ="
            sym_tot_local = True
        out = generate_answer(model, prompt)
        pred = parse_answer(out, len(prompt))
        ok = pred == r
        total += 1
        if use_en:
            en_tot += 1; en_ok += int(ok)
        else:
            sym_tot += 1; sym_ok += int(ok)
    return {
        "math_acc": (sym_ok + en_ok) / max(1, total),
        "math_acc_symbolic": sym_ok / max(1, sym_tot),
        "math_acc_english": en_ok / max(1, en_tot),
        "math_total": float(total),
    }


# Gate: arithmetic is learned only when answers are reliably correct.
MATH_TARGET = 0.95


def passed_math(metrics):
    return metrics.get("math_acc", 0.0) >= MATH_TARGET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="daycare_literate/math.txt")
    ap.add_argument("--max-operand", type=int, default=9)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--hold-out", type=float, default=0.2)
    args = ap.parse_args()
    train, test = split_problems(args.max_operand, args.hold_out)
    text = build_corpus(train, repeats=args.repeats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} chars): train {len(train)} / "
          f"held-out {len(test)} problems, digits + English")


if __name__ == "__main__":
    main()
