"""Numerical correctness tests for the Metatron geometric learner.

These are the "self-generated tests" the architecture doc requires: teacher
opinion is never accepted without an executable check. Run with:

    python -m pytest tests/ -q
"""
from __future__ import annotations

import numpy as np
import pytest

from metatron import MetatronV2, SCALES, verify_capabilities, make_batches, train
from metatron.metatron_v2 import (
    Config, GeometricAttention, SwiGLU, Param, _ln_fwd, _ln_bwd,
)


def tiny_cfg():
    return Config(name="fd", dim=16, n_modules=3, n_heads=2, n_layers=1,
                  vocab_size=64, context_length=16, batch_size=1)


def grad_check(params, loss_fn, probes, eps=1e-4, rel_tol=0.08, abs_tol=2e-2):
    """Assert analytic grads match central finite differences.

    ``loss_fn()`` MUST start by zeroing every parameter's grad and return the
    scalar loss; ``probes`` is a list of (Param, index-tuple-or-int).
    """
    for p, idx in probes:
        loss_fn()
        ana = float(p.grad[idx])
        orig = float(p.data[idx])
        p.data[idx] = orig + eps
        lp = loss_fn()
        p.data[idx] = orig - eps
        lm = loss_fn()
        p.data[idx] = orig
        num = (lp - lm) / (2 * eps)
        assert ana == pytest.approx(num, rel=rel_tol, abs=abs_tol), (
            f"grad mismatch at {p.data.shape}[{idx}]: analytic={ana:.6f} "
            f"numeric={num:.6f}")
        del lp, lm


# --------------------------------------------------------------------- units
def test_layernorm_scale_gradient():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32).astype(np.float32)
    s = Param(np.ones(32, np.float32))
    g = rng.standard_normal(32).astype(np.float32)

    def loss():
        s.grad.fill(0.0)
        y, cache = _ln_fwd(x, s.data)
        _ln_bwd(g, cache, s.grad)
        return float((y * g).sum())

    grad_check([s], loss, [(s, 7)], eps=1e-4, rel_tol=0.03)


def test_swiglu_gradients():
    rng = np.random.default_rng(1)
    ffn = SwiGLU(32)
    x = rng.standard_normal(32).astype(np.float32)
    dy = rng.standard_normal(32).astype(np.float32)
    params = list(ffn.w1.params()) + list(ffn.w2.params()) + list(ffn.w3.params())

    def loss():
        for p in params:
            p.grad.fill(0.0)
        y, cache = ffn.forward(x)
        ffn.backward(dy, cache)
        return float((y * dy).sum())

    grad_check(params, loss, [
        (ffn.w1.W, (2, 5)), (ffn.w2.W, (3, 6)), (ffn.w3.W, (7, 9)),
    ], eps=1e-4, rel_tol=0.05)


def test_attention_gradients():
    rng = np.random.default_rng(2)
    cfg = Config(name="t", dim=32, n_modules=5, n_heads=4, n_layers=1,
                 vocab_size=64)
    attn = GeometricAttention(cfg)
    states0 = [rng.standard_normal(32).astype(np.float32) for _ in range(5)]
    x0 = rng.standard_normal(32).astype(np.float32)
    dy = rng.standard_normal(32).astype(np.float32)
    params = list(attn.params())

    def states():
        return [s.copy() for s in states0]

    def loss():
        for p in params:
            p.grad.fill(0.0)
        y, cache = attn.forward(x0, states())
        dS = np.zeros((5, 32), np.float32)
        attn.backward(dy, cache, dS)
        return float((y * dy).sum()) + 0.0 * float(dS.sum())

    grad_check(params, loss, [
        (attn.Wq.W, (2, 5)), (attn.Wk.W, (1, 7)), (attn.Wv.W, (3, 8)),
        (attn.Wo.W, (4, 9)), (attn.Wo.b, 11),
        (attn.router.W, (2, 6)), (attn.router.b, 3),
    ], eps=1e-4, rel_tol=0.05)


# --------------------------------------------------------------- full model
def test_full_bptt_matches_finite_differences():
    """End-to-end BPTT vs central differences on a minimal recurrent model."""
    m = MetatronV2(tiny_cfg(), seed=3)
    ids = m.encode("abc", add_bos=True, add_eos=True)

    def loss():
        m.zero_grad()
        return m.loss(ids)

    params = m.all_params()
    grad_check(params, loss, [
        (m.modules[0][0].attn.Wo.b, 5),
        (m.modules[2][0].attn.Wo.b, 7),
        (m.modules[1][0].ffn.w1.W, (2, 4)),
        (m.modules[1][0].ffn.w3.W, (3, 2)),
        (m.modules[0][0].n1, 6),
        (m.modules[1][0].n2, 2),
        (m.embed, (ids[0], 3)),
        (m.pos, (1, 4)),
        (m.final_norm, 9),
        (m.modules[0][0].attn.router.W, (1, 3)),
        (m.modules[2][0].attn.Wv.W, (2, 2)),
        (m.head.b, 10),
    ], eps=1e-3, rel_tol=0.10, abs_tol=0.15)


def test_gradients_reach_every_parameter():
    m = MetatronV2(tiny_cfg(), seed=4)
    m.zero_grad()
    m.loss(m.encode("geometric flower", add_bos=True, add_eos=True))
    dead = [i for i, p in enumerate(m.all_params())
            if not np.isfinite(p.grad).all() or float(np.abs(p.grad).sum()) == 0.0]
    assert not dead, f"{len(dead)} parameters receive no gradient"


def test_overfit_short_sequence():
    m = MetatronV2(tiny_cfg(), seed=5)
    ids = m.encode("metatron is geometric", add_bos=True, add_eos=True)
    losses = []
    for _ in range(40):
        losses.append(m.loss(ids))
        m.step(0.01)
    assert np.isfinite(losses[-1])
    assert losses[-1] < 0.5 * losses[0]


def test_all_capabilities_verified():
    caps = verify_capabilities(MetatronV2(tiny_cfg(), seed=6))
    failed = [k for k, v in caps.items() if not v]
    assert not failed, f"capability failures: {failed}"


def test_training_reduces_window_loss():
    """Real learning signal: loss after a few epochs is strictly lower."""
    m = MetatronV2(SCALES["nano"], seed=7)
    text = ("the flower of life geometry circles sacred metatron "
            "learning patterns networks ") * 4
    windows = make_batches(text, m, m.cfg.context_length, m.cfg.batch_size)
    seq = windows[0][0]
    before = m.loss(seq)
    train(m, text, epochs=3, out_dir="/tmp/metatron_test_ckpts", verbose=False)
    after = m.loss(seq)
    assert after < before, f"no learning: {before:.3f} -> {after:.3f}"


def test_generation_is_text_and_logits_finite():
    m = MetatronV2(SCALES["nano"], seed=8)
    out = m.generate("the flower", max_new=20, temperature=0.7)
    assert isinstance(out, str) and len(out) > 0
    logits = m.forward(m.encode("test", add_bos=True))
    assert np.isfinite(logits).all()
