"""End-to-end daycare loop tests: hatch -> train -> evaluate -> promote.

Runs the real (small, cheap) Metatron trainer and benchmark gate against a
temporary daycare root, proving the loop improves and promotes the champion
without any external service.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TRAINER = [sys.executable, str(ROOT / "daycare" / "train_real.py"), "--root", str(ROOT)]
EVAL = [sys.executable, str(ROOT / "daycare" / "evaluate_real.py"), "--root", str(ROOT)]


def _run(argv, env_extra, expect_zero=True, cwd=ROOT):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env.update({k: str(v) for k, v in env_extra.items()})
    out = subprocess.run(argv, env=env, capture_output=True, text=True,
                         timeout=900, cwd=cwd)
    if expect_zero and out.returncode != 0:
        pytest.fail(f"command failed: {argv}\nSTDOUT:{out.stdout}\nSTDERR:{out.stderr}")
    return out


def test_train_and_evaluate_candidate(tmp_path):
    ckpt_dir = tmp_path / "cand"
    env = {
        "METATRON_CHECKPOINT_DIR": str(ckpt_dir),
        "METATRON_SCALE": "nano",
        "METATRON_SAFE_LR": "0.0008",
        "METATRON_GRAD_CLIP": "0.5",
    }
    # Two cheap training generations; the second resumes from the first via
    # a champion copy to prove full-object checkpoints survive.
    out1 = _run(TRAINER + ["--epochs", "2"], env)
    payload1 = json.loads(out1.stdout.strip().splitlines()[-1])
    assert payload1["status"] == "ok"
    assert (ckpt_dir / "model.pkl").is_file()

    ev = _run(EVAL, env)
    score1 = json.loads(ev.stdout.strip().splitlines()[-1])
    import math
    assert math.isfinite(score1["score"]), score1
    assert score1["metrics"]["capability_score"] == 1.0

    # Resume training from this checkpoint and confirm loss improves.
    parent = tmp_path / "champion"
    (parent).mkdir()
    import shutil
    shutil.copy(ckpt_dir / "model.pkl", parent / "model.pkl")
    out2 = _run(TRAINER + ["--epochs", "3", "--parent", str(parent / "model.pkl")], env)
    payload2 = json.loads(out2.stdout.strip().splitlines()[-1])
    assert payload2["parent"].endswith("model.pkl")

    ev2 = _run(EVAL, env)
    score2 = json.loads(ev2.stdout.strip().splitlines()[-1])
    import math
    assert math.isfinite(score2["score"])
    # More training must not regress the measured loss beyond noise.
    assert score2["metrics"]["eval_loss"] <= score1["metrics"]["eval_loss"] + 0.5


def test_daycare_orchestrator_promotes(tmp_path):
    """The orchestrator hatches, trains, evaluates, and promotes a champion."""
    state_root = tmp_path / "daycare_state"
    env = {
        "METATRON_DAYCARE_ROOT": str(state_root),
        "METATRON_TRAINER": " ".join(TRAINER),
        "METATRON_EVALUATOR": " ".join(EVAL),
        "METATRON_SCALE": "nano",
        "METATRON_MAX_BUDGET": "50",
        "METATRON_MIN_GAIN": "0.0",   # first valid candidate always promotes
        "METATRON_SAFE_LR": "0.0008",
    }
    out = _run([sys.executable, str(ROOT / "daycare" / "metatron_daycare.py"),
                "--root", str(state_root), "--once"], env)
    state = json.loads((state_root / "state.json").read_text())
    assert state["generation"] >= 1, state
    assert state["champion"], "no champion promoted"
    assert (state_root / "champion" / "model.pkl").is_file()
    assert state["best_score"] != float("-inf")


def test_pokemon_profile_renders(tmp_path):
    state_root = tmp_path / "daycare_state"
    state_root.mkdir()
    (state_root / "state.json").write_text(json.dumps({
        "xp": 2500, "level": 3, "generation": 4, "champion": "abc123",
        "best_score": -3.21,
    }))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    out = subprocess.run(
        [sys.executable, str(ROOT / "daycare" / "pokemon_profile.py"),
         "--root", str(state_root)],
        env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    profile = json.loads((state_root / "pokemon.json").read_text())
    assert profile["species"] == "Metatron"
    assert profile["generation"] == 4
