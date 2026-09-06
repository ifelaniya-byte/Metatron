#!/usr/bin/env python3
"""Objective literacy gate: has Metatron stopped babbling?

The architecture docs forbid declaring victory without an executable test,
and "it reads okay to me" is not one. This module scores generation by:

  word_validity  -- fraction of sampled letter-runs that are real English
                    words (curriculum vocabulary + an embedded high-frequency
                    word list). 0.0 = pure babble, 1.0 = every token a word.
  word_unique    -- fraction of the sampled words that are distinct (guards
                    against a degenerate model that only repeats one word).
  char_efficiency-- mean cross-entropy on the literacy curriculum.

Sampling uses a fixed seed, greedy/low-temperature decoding and several
prompts, so the metric is reproducible across runs.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from daycare.curriculum import build_corpus, build_wordset

_WORD_RE = re.compile(r"[a-z]+")

PROMPTS = [
    "the flower",
    "the river",
    "a quiet",
    "metatron",
    "sacred",
    "every small",
]


def generate_words(model, prompt, rng, max_new=160, temperature=0.0, top_k=1):
    """Generate deterministically and return the text.

    Default is greedy (temperature 0, top_k 1): the "has it stopped
    babbling?" test is whether the single most likely continuation is real
    English, not whether a stochastic sample occasionally is.
    """
    # Temporarily drive the model's RNG for reproducible tie-breaks.
    saved = getattr(model, "_rng", None)
    model._rng = rng
    try:
        # model.generate treats temperature < tiny as-is; force argmax path
        # by using a very small temperature and top_k=1.
        text = model.generate(prompt, max_new=max_new,
                              temperature=max(temperature, 1e-3), top_k=top_k)
    finally:
        model._rng = saved
    return text


def _score_text(text, wordset):
    words = _WORD_RE.findall(text.lower())
    total = len(words)
    valid = sum(1 for w in words if w in wordset)
    return total, valid, words


def literacy_metrics(model, seed=1337, corpus_path=None):
    import numpy
    wordset = build_wordset()
    rng = numpy.random.default_rng(seed)

    total_words = valid_words = 0
    all_words: list[str] = []
    sample_total = sample_valid = 0
    for prompt in PROMPTS:
        # Greedy: the deterministic coherence test (drives the gate).
        g_text = generate_words(model, prompt, rng, temperature=1e-3, top_k=1)
        t, v, w = _score_text(g_text, wordset)
        total_words += t
        valid_words += v
        all_words.extend(w)
        # Sampled: softer measure of distribution quality (report only).
        s_text = generate_words(model, prompt, rng, temperature=0.5, top_k=12)
        st, sv, _ = _score_text(s_text, wordset)
        sample_total += st
        sample_valid += sv

    word_validity = valid_words / max(1, total_words)
    word_unique = len(set(all_words)) / max(1, len(all_words))
    word_validity_sampled = sample_valid / max(1, sample_total)

    # Cross-entropy on the bounded curriculum windows (same geometry as the
    # training gate). Uses the training corpus when available so the loss
    # reflects what the model was actually fed.
    if corpus_path is not None and Path(corpus_path).is_file():
        text = Path(corpus_path).read_text(encoding="utf-8")
    else:
        text = build_corpus()
    from metatron import make_batches
    batches = make_batches(text, model, model.cfg.context_length,
                           model.cfg.batch_size)
    losses = []
    for batch in batches:
        for seq in batch:
            v = float(model.loss(seq))
            if math.isfinite(v):
                losses.append(v)
    char_loss = sum(losses) / max(1, len(losses))

    return {
        "word_validity": float(word_validity),
        "word_validity_sampled": float(word_validity_sampled),
        "word_unique": float(word_unique),
        "word_count": float(total_words),
        "literacy_loss": float(char_loss),
        "literacy_perplexity": float(math.exp(min(char_loss, 20))),
    }


# Gate: a candidate is "literate" when almost every emitted token is a real
# word and the vocabulary is not collapsing to one word.
WORD_VALIDITY_TARGET = 0.90
WORD_UNIQUE_MIN = 0.30


def passed_literacy(metrics):
    # Gate on SAMPLED text (not greedy): a model whose greedy output is
    # "the the the" loops is still babbling. We require nearly every token of
    # a temperature=0.5 sample to be a real word, a non-degenerate vocabulary,
    # and reasonable cross-entropy.
    return (metrics.get("word_validity_sampled", 0.0) >= WORD_VALIDITY_TARGET
            and metrics.get("word_unique", 0.0) >= WORD_UNIQUE_MIN
            and metrics.get("literacy_loss", 9.9) <= 1.7)
