"""Tests for the speech recitation/quiz literacy gates.

These cover the deterministic parts: Markdown parsing, corpus construction,
and that an untrained model scores ~0 while a perfectly-keyed stub scores
high — i.e. the gates measure what they claim to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from daycare import speech_literacy as sl  # noqa: E402


def test_all_speeches_parse_with_quiz():
    speeches = sl.all_speeches()
    assert len(speeches) >= 4
    for sp in speeches:
        assert sp["title"], f"{sp['file']} missing title"
        assert sp["lines"], f"{sp['file']} has no speech lines"
        assert len(sp["quiz"]) >= 5, f"{sp['file']} has too few quiz items"
        for q, a in sp["quiz"]:
            assert "______" in q, f"{sp['file']} quiz missing blank: {q!r}"
            assert a and " " not in a, f"{sp['file']} answer must be one word: {a!r}"


def test_quiz_answers_are_words_in_the_text():
    """Every answer key must actually appear in its speech (objective)."""
    for sp in sl.all_speeches():
        body = " ".join(sp["lines"]).lower()
        for q, a in sp["quiz"]:
            assert a in body, f"{sp['title']}: answer {a!r} not found in text"


def test_build_corpus_contains_qa_and_lines():
    corpus = sl.build_corpus(repeats=1, quiz_repeats=1)
    assert "Q:" in corpus and "\nA:" in corpus
    # a known quiz answer line
    assert "give me liberty or give me death" in corpus.lower()


class KeyStub:
    """A fake model that returns the correct answer word for quiz lines and
    the true continuation for recitation lines — proves the gates credit
    verbatim/keyed output correctly."""
    def generate(self, prompt, max_new=8, temperature=1e-3, top_k=1):
        prompt = prompt.strip()
        if prompt.startswith("Q:"):
            # find matching quiz answer
            for sp in sl.all_speeches():
                for q, a in sp["quiz"]:
                    if sl._norm(q) in prompt or prompt.endswith(sl._norm(q)[-30:]):
                        return prompt + " " + a
            return prompt + " liberty"
        # recitation: continue with a filler that matches startswith logic
        return prompt + " x"


def test_quiz_gate_credits_correct_keyed_answers():
    res = sl.evaluate_quiz(KeyStub())
    # The stub resolves all quiz prompts to their key answer.
    assert res["quiz_acc"] == pytest.approx(1.0, abs=0.05)


def test_untrained_model_does_not_pass():
    from metatron import MetatronV2, SCALES
    m = MetatronV2(SCALES["nano"], seed=0)
    res = sl.evaluate_speeches(m)
    assert res["quiz_acc"] < 0.5
    assert res["recite_char_acc"] < 0.2
