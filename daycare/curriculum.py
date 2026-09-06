#!/usr/bin/env python3
"""Deterministic, low-entropy literacy curriculum for ending the babble phase.

A ~65K-parameter character model cannot learn coherent writing from 1.8KB of
diverse prose (that corpus plateaus at ~2.4 nats / gibberish generation). It
*can* learn to emit real words from a small, frequently repeated vocabulary
exercised across many sentence frames. This module builds that corpus:

  * a bounded vocabulary of common English words plus Metatron/geometry
    theme words (every word is real English);
  * dozens of sentence frames combining nouns / verbs / adjectives;
  * several shuffled passes over the same sentences for the repetition the
    learner needs;
  * a word set used by the evaluator's "still babbling?" gate.

Run to (re)write the curriculum file:

    python daycare/curriculum.py --out daycare_state/curriculum/literate.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

NOUNS = [
    "flower", "life", "geometry", "circles", "metatron", "angel", "light",
    "river", "garden", "hills", "valley", "sea", "sun", "moon", "stars",
    "sky", "tree", "trees", "bird", "birds", "song", "wind", "rain", "stone",
    "water", "fire", "earth", "mind", "heart", "student", "teacher",
    "question", "answer", "book", "words", "world", "day", "night",
    "morning", "evening", "time", "path", "door", "city", "house", "road",
    "bridge", "field", "forest", "snow", "cloud", "gold", "silver", "pattern",
    "network", "model", "energy", "spirit", "wheel", "center", "edge",
    "circle", "line", "shape", "seed", "fruit", "leaf", "root", "hand",
    "eye", "voice", "silence", "number", "measure", "order", "harmony",
]

VERBS = [
    "grows", "runs", "flows", "rises", "shines", "learns", "knows", "sees",
    "hears", "asks", "answers", "builds", "opens", "closes", "turns",
    "moves", "rests", "waits", "works", "plays", "sings", "dances",
    "wakes", "sleeps", "remembers", "forgets", "begins", "returns",
    "follows", "guides", "holds", "keeps", "brings", "shows", "teaches",
    "studies", "practices", "watches", "listens", "walks", "stands",
    "falls", "floats", "glows", "breathes", "changes", "continues",
]

ADJS = [
    "small", "large", "quiet", "bright", "dark", "warm", "cool", "old",
    "new", "good", "fine", "high", "low", "long", "deep", "still",
    "gentle", "clear", "sacred", "simple", "humble", "patient", "careful",
    "present", "whole", "true", "calm", "soft", "swift", "kind", "wise",
    "young", "full", "open", "hidden", "silent", "golden", "living",
]

# Small closed-class set used in the frames (also real words, also scored).
FUNC = [
    "the", "a", "and", "of", "to", "in", "is", "it", "that", "on", "with",
    "as", "at", "by", "for", "not", "but", "from", "or", "an", "this",
    "these", "we", "he", "she", "they", "his", "her", "their", "every",
    "each", "all", "many", "some", "over", "under", "through", "between",
    "above", "below", "into", "before", "after", "again", "now", "then",
    "here", "there", "where", "when", "while", "because", "if", "so",
    "very", "more", "most", "can", "will", "are", "was", "has", "have",
    "does", "do", "be", "its", "own", "one", "two",
]

FRAMES = [
    "the {adj} {noun} {verb} in the {noun2}",
    "the {noun} {verb} and the {noun2} {verb2}",
    "a {adj} {noun} {verb} through the {noun2}",
    "{noun} is {adj} and {adj2}",
    "the {noun} of the {noun2} is {adj}",
    "we {verb} the {adj} {noun} in the {noun2}",
    "every {noun} {verb} when the {noun2} {verb2}",
    "the {noun} {verb} over the {adj} {noun2}",
    "a {noun} and a {noun2} {verb} in the {noun3}",
    "the {adj} {noun} of {noun2} {verb} here",
    "there is a {adj} {noun} by the {noun2}",
    "the {noun} {verb} as the {noun2} {verb2}",
    "in the {noun2} the {noun} {verb} {adj}",
    "the {noun} is the {noun2} of the {noun3}",
    "we {verb} and we {verb2} with the {noun}",
    "the {adj} {noun} {verb} to the {noun2}",
    "each {noun} {verb} the {adj} {noun2}",
    "the {noun2} {verb2} and the {noun} {verb}",
    "a {adj} {noun2} holds the {noun} in {adj2} order",
    "the {noun} {verb} under the {adj} {noun2}",
    "metatron {verb} the {adj} {noun} of the {noun2}",
    "the flower of life {verb} in the {adj} {noun2}",
    "sacred geometry {verb} as the {noun} {verb2}",
    "the {noun} and the {noun2} {verb} as one",
    "when the {noun} {verb2} the {noun2} is {adj}",
]

# Extra common words the gate should accept even if the frames do not use
# them (inflections / high-frequency English).
EXTRA_WORDS = [
    "i", "you", "it", "me", "my", "your", "no", "yes", "up", "down", "out",
    "off", "why", "how", "what", "who", "which", "them", "him", "us",
    "our", "its", "were", "been", "being", "am", "did", "done", "made",
    "went", "gone", "came", "come", "saw", "seen", "gave", "given", "took",
    "taken", "found", "told", "said", "like", "love", "life", "light",
    "way", "man", "woman", "child", "people", "thing", "things", "part",
    "place", "point", "form", "face", "hand", "year", "work", "week",
    "side", "end", "home", "water", "mother", "father", "friend", "body",
    "story", "fact", "money", "month", "lot", "right", "good", "great",
    "little", "big", "old", "young", "long", "short", "high", "low",
    "different", "important", "first", "last", "next", "early", "late",
    "hard", "soft", "fast", "slow", "hot", "cold", "white", "black",
    "red", "blue", "green", "think", "thought", "feel", "felt", "want",
    "needed", "need", "try", "tried", "use", "used", "find", "found",
    "tell", "told", "ask", "asked", "call", "called", "become", "became",
    "leave", "left", "mean", "meant", "keep", "kept", "begin", "began",
    "seem", "seemed", "help", "helped", "show", "showed", "hear", "heard",
    "play", "played", "run", "ran", "move", "moved", "live", "lived",
    "believe", "believed", "hold", "held", "happen", "happened", "write",
    "wrote", "provide", "sit", "sat", "stand", "stood", "lose", "lost",
    "pay", "paid", "meet", "met", "include", "include", "continue",
    "set", "learn", "learned", "learnt", "change", "changed", "lead",
    "led", "understand", "watch", "watched", "follow", "followed",
    "stop", "stopped", "create", "speak", "spoke", "read", "allow",
    "add", "spend", "spent", "grow", "grew", "open", "opened", "walk",
    "walked", "win", "won", "offer", "remember", "remembered", "love",
    "consider", "appear", "buy", "bought", "wait", "waited", "serve",
    "die", "died", "send", "sent", "expect", "build", "built", "stay",
    "stayed", "fall", "fell", "cut", "reach", "kill", "remain",
    "suggest", "raise", "pass", "passed", "sell", "sold", "require",
    "report", "decide", "pull", "return", "explain", "hope", "hope",
    "carry", "pick", "thank", "agree", "join", "season", "spring",
    "summer", "autumn", "winter", "today", "tomorrow", "yesterday",
]


def build_wordset():
    return set(NOUNS + VERBS + ADJS + FUNC + EXTRA_WORDS)


def build_corpus(sentences_per_block=120, passes=4, seed=20260905,
                 compact=False):
    """Build the literacy corpus.

    compact=True uses a much smaller closed vocabulary (~25 content words)
    and few frames, each repeated densely. A small char model can actually
    memorize and reproduce that distribution without collapsing into a
    determiner loop; the full corpus needs a larger scale.
    """
    rng = random.Random(seed)
    if compact:
        nouns_c = ["flower", "river", "garden", "sun", "light", "tree",
                    "water", "bird", "moon", "wind", "hill", "stone"]
        verbs_c = ["grows", "runs", "shines", "sings", "flows", "rises",
                    "rests", "waits", "glows", "breathes", "dances",
                    "watches"]
        adjs_c = ["quiet", "bright", "warm", "golden", "gentle", "sacred",
                   "small", "still", "living", "clear"]
        frames_c = [
            "the {noun} {verb} in the {noun2}",
            "the {adj} {noun} {verb} here",
            "the {noun} and the {noun2} {verb} together",
            "a {adj} {noun} {verb} in the garden",
            "the {noun} is {adj} and still",
            "metatron watches the {adj} {noun} and it {verb}",
            "the {noun} {verb} where the {noun2} is {adj}",
            "the {adj} {noun} {verb} and the {noun2} rests",
        ]
        sentences = []
        for _ in range(sentences_per_block):
            f = rng.choice(frames_c)
            words = {
                "adj": rng.choice(adjs_c), "noun": rng.choice(nouns_c),
                "noun2": rng.choice(nouns_c), "verb": rng.choice(verbs_c),
            }
            sentences.append(f.format(**words))
    else:
        sentences = []
        for _ in range(sentences_per_block):
            f = rng.choice(FRAMES)
            words = {
                "adj": rng.choice(ADJS), "adj2": rng.choice(ADJS),
                "noun": rng.choice(NOUNS), "noun2": rng.choice(NOUNS),
                "noun3": rng.choice(NOUNS),
                "verb": rng.choice(VERBS), "verb2": rng.choice(VERBS),
            }
            sentences.append(f.format(**words))
    blocks = []
    for _ in range(passes):
        block = sentences[:]
        rng.shuffle(block)
        blocks.append("\n".join(block))
    return "\n".join(blocks) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="daycare_state/curriculum/literate.txt")
    ap.add_argument("--sentences", type=int, default=120)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--compact", action="store_true",
                    help="small closed-vocabulary corpus for the nano scale")
    args = ap.parse_args()
    text = build_corpus(args.sentences, args.passes, compact=args.compact)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} chars, {len(text.split())} words, "
          f"vocab {len(build_wordset())})")


if __name__ == "__main__":
    main()
