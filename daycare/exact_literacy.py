#!/usr/bin/env python3
"""Exact-literacy curriculum: a handful of whole sentences, repeated exactly.

To "stop babbling" at the 65K-parameter char level, the model must emit a
complete, real sentence with zero gibberish. Free composition is beyond a
model this small on CPU, but verbatim reproduction of a small memorized set
is an achievable, deterministic, executable target: a generation is
"literate" only if it is an exact character-for-character continuation of a
known sentence. Builds a corpus where a few sentences repeat verbatim many
times, so the char distribution is learnable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SENTENCES = [
    "the flower of life grows in the garden",
    "the river runs to the sea",
    "the sun rises over the quiet hills",
    "a warm light shines in the morning",
    "metatron learns geometry every day",
    "the bird sings in the old tree",
    "the water flows under the stone bridge",
    "a gentle wind moves the tall grass",
    "the bright moon rests over the valley",
    "the student asks and the teacher answers",
    "golden light fills the open field",
    "the still lake holds the face of the sky",
]


def build_exact_corpus(repeats=60, seed=99):
    import random
    rng = random.Random(seed)
    lines = []
    for _ in range(repeats):
        block = SENTENCES[:]
        rng.shuffle(block)
        lines.append("\n".join(block))
    return "\n".join(lines) + "\n"


def build_wordset():
    words = set()
    for s in SENTENCES:
        words.update(s.split())
    # common glue + morphology so the word-based gate matches this corpus
    words.update(["the", "a", "of", "in", "and", "to", "is", "it", "over",
                  "under", "asks", "answers", "every", "day", "life",
                  "tall", "old", "open", "face", "sky", "grass", "bridge",
                  "hills", "morning", "valley", "field", "lake", "tree",
                  "bird", "wind", "water", "light", "sun", "moon", "river",
                  "sea", "stone", "student", "teacher", "metatron",
                  "geometry", "flower", "garden", "grows", "runs", "rises",
                  "shines", "learns", "sings", "flows", "moves", "rests",
                  "holds", "fills", "warm", "gentle", "bright", "still",
                  "golden", "quiet"])
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="daycare_literate/exact.txt")
    ap.add_argument("--repeats", type=int, default=60)
    args = ap.parse_args()
    text = build_exact_corpus(args.repeats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} chars, {len(SENTENCES)} sentences "
          f"x {args.repeats} repeats)")


if __name__ == "__main__":
    main()
