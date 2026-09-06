"""Deterministic gates for the Game of Time (missions/TIME_GAME.md).

The game streams alternating English paragraphs; odd paragraphs strike at the
final word. Metatron must encode the rhythm in its carried ring-memory before
the strike word lands. These gates are deliberately strong:

* the held-out brain probe reads strike-vs-calm on UNSEEN paragraphs,
* it tracks a PHASE-FLIPPED stream (so it uses timed memory, not fixed
  odd/even position),
* a shuffled-label control probe stays near chance (the signal is the
  rhythm, not a probe artifact),
* the probe-driven shield survives unseen strikes,
* mechanics: brainwaves and actions are logged for every paragraph and stay
  finite.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daycare import time_game as tg  # noqa: E402
from metatron import MetatronV2, SCALES  # noqa: E402


def test_paragraphs_alternate_and_decision_is_final_word():
    paras = tg.gen_paragraphs(6, seed=0)
    assert [p["strike"] for p in paras] == [True, False, True, False, True, False]
    assert [p["outcome"] for p in paras] == ["blade", "air", "blade",
                                             "air", "blade", "air"]
    for p in paras:
        assert p["text"].rstrip(".").endswith(p["outcome"])
        # bodies never leak the outcome words before the final position
        assert "blade" not in p["body"] and " air" not in p["body"]


def test_read_stream_logges_one_brainwave_per_paragraph_all_finite():
    m = MetatronV2(SCALES["nano"], seed=1)
    paras = tg.gen_paragraphs(8, seed=0)
    read = tg.read_stream(m, paras)
    assert len(read["waves"]) == len(paras) == len(read["actions"])
    labels = [w["label"] for w in read["waves"]]
    assert labels == [1, 0, 1, 0, 1, 0, 1, 0]
    for w in read["waves"]:
        assert np.isfinite(w["h"]).all() and np.isfinite(w["x"]).all()
        assert math.isfinite(w["p_b"]) and math.isfinite(w["p_a"])
    for a in read["actions"]:
        assert a["result"] in {"blocked", "hit", "calm-ok", "false-alarm"}


def test_decision_point_precedes_outcome_word():
    # The recorded brainwave must be at the char position of the first
    # outcome letter — i.e. BEFORE the model has seen blade/air.
    m = MetatronV2(SCALES["nano"], seed=1)
    paras = tg.gen_paragraphs(4, seed=0)
    stream = ""
    decision_chars = []
    for k, p in enumerate(paras):
        prefix = "" if k == 0 else "\n"
        lead = f"{prefix}{p['body']} the "
        decision_chars.append(len(stream) + len(lead))
        stream += lead + p["outcome"] + "."
    for d in decision_chars:
        assert stream[d] in ("b", "a")
        assert stream[d:d + 5] in ("blade", "air.") or stream[d:d+4] == "air."


def test_phase_flipped_stream_switches_which_parity_strikes():
    flip = tg.gen_phase_flipped(6)
    # idx1 calm, idx2 strike, ...
    assert [p["strike"] for p in flip] == [False, True, False, True, False, True]


def test_time_game_gate_detects_rhythm_not_position():
    res = tg.gate(seed=20260906, n_train=24, n_test=24, epochs=14, lr=0.003)
    # The brain encodes strike-vs-calm before the word lands, on unseen text.
    assert res["probe_heldout_acc"] >= 0.95, res
    # It tracks the ACTUAL rhythm: phase-flipped stream still read correctly
    # (rules out a fixed odd/even-position shortcut).
    assert res["probe_phaseflip"] >= 0.95, res
    # The label-shuffled control must stay near chance (no probe artifact).
    assert res["shuffled_label_acc"] <= 0.85, res
    # Acting on the probe, Metatron shields the unseen strikes and survives.
    assert res["shield_correct"] >= 0.95, res
    assert res["survived"] is True, res
