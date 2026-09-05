#!/usr/bin/env python3
"""Deterministic benchmark gate for a persisted Metatron candidate.

Runs the capability suite on a deep copy (evaluation must never mutate the
candidate) and measures mean next-token loss over the same bounded-context
windows used by training. Emits one JSON line:

    {"score": float, "metrics": {...}}

NaN/Inf loss or any failed capability rejects the candidate (score = -inf).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_from_path(path: Path):
    import importlib.util
    name = "metatron_v2_real"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_metatron(root: Path):
    """Prefer the vendored package; fall back to workspace copies / zips."""
    try:
        from metatron import metatron_v2 as mod
        return mod
    except ImportError:
        pass

    loose = [p for p in root.rglob("metatron_v2.py")
             if "_workspace_source" not in p.parts and "_metatron_source" not in p.parts]
    if loose:
        return _load_from_path(loose[0])

    for pattern, dest in (("*grok-workspace.zip", root / "_workspace_source"),
                          ("Metatron_Full_System.zip", root / "_metatron_source")):
        for z in root.rglob(pattern):
            try:
                dest.mkdir(exist_ok=True)
                with zipfile.ZipFile(z) as f:
                    f.extractall(dest)
                for nested in list(dest.rglob("Metatron_Full_System.zip")):
                    with zipfile.ZipFile(nested) as nf:
                        nf.extractall(dest / "release")
                found = list(dest.rglob("metatron_v2.py"))
                if found:
                    return _load_from_path(found[0])
            except zipfile.BadZipFile:
                continue
    raise FileNotFoundError("metatron implementation not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--text", default="")
    args = ap.parse_args()

    path = args.checkpoint or os.path.join(
        os.getenv("METATRON_CHECKPOINT_DIR", ""), "model.pkl")
    if not path or not os.path.isfile(path):
        raise SystemExit("candidate checkpoint missing: " + str(path))

    mod = load_metatron(Path(args.root).resolve())
    with open(path, "rb") as f:
        model = pickle.load(f)

    benchmark_error = ""
    try:
        caps = mod.verify_capabilities(copy.deepcopy(model))
    except Exception as e:  # a candidate that crashes verification is rejected
        caps = {"benchmark_exception": False}
        benchmark_error = str(e)

    passed = sum(bool(v) for v in caps.values())
    total = len(caps)

    text = (Path(args.text).read_text(encoding="utf-8")
            if args.text and Path(args.text).is_file() else mod.DEFAULT_TEXT)

    # Same bounded context windows as training; evaluating the whole stream in
    # one pass makes this recurrent geometry numerically meaningless.
    batches = mod.make_batches(text, model, model.cfg.context_length,
                               model.cfg.batch_size)
    losses = []
    finite = True
    for batch in batches:
        for seq in batch:
            try:
                value = float(model.loss(seq))
            except Exception as e:
                finite = False
                benchmark_error = benchmark_error or f"loss: {e}"
                continue
            if not math.isfinite(value):
                finite = False
                continue
            losses.append(value)

    loss = sum(losses) / max(1, len(losses))
    finite = finite and bool(losses) and math.isfinite(loss)
    capability_score = passed / max(1, total)
    score = -loss if (passed == total and finite) else float("-inf")

    metrics = {f"cap_{k}": float(bool(v)) for k, v in caps.items()}
    metrics.update({
        "capability_score": capability_score,
        "eval_loss": loss,
        "eval_windows": float(len(losses)),
        "eval_perplexity": (min(1e12, math.exp(min(loss, 27)))
                            if finite else float("inf")),
    })
    if benchmark_error:
        metrics["benchmark_exception"] = 1.0
    print(json.dumps({"score": score, "metrics": metrics}))


if __name__ == "__main__":
    main()
