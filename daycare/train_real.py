#!/usr/bin/env python3
"""Real Metatron trainer adapter with full-state, shutdown-safe checkpoints.

Loads the actual learnable Metatron implementation (vendored ``metatron``
package first, then any ``metatron_v2.py`` discovered in the workspace, then
the archived full-system zip as a last resort), starts from the current
champion when a checkpoint exists, trains on the bounded-context curriculum,
and pickles the *entire* model object (weights + Adam moments + tokenizer)
into the candidate checkpoint directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("metatron_v2_real", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract_zip(outer: Path, dest: Path) -> bool:
    try:
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(outer) as zf:
            zf.extractall(dest)
        return True
    except zipfile.BadZipFile:
        return False


def load_metatron(root: Path):
    """Return the Metatron implementation module, preferring vendored code."""
    # 1. Vendored package in the repository (source of truth).
    try:
        import metatron  # noqa: F401
        from metatron import metatron_v2 as mod
        return mod
    except ImportError:
        pass

    # 2. Loose metatron_v2.py somewhere in the workspace.
    candidates = list(root.rglob("metatron_v2.py"))
    candidates = [c for c in candidates if "_workspace_source" not in c.parts
                  and "_metatron_source" not in c.parts]
    if candidates:
        return _load_from_path(candidates[0])

    # 3. Archived full-system zips (workspace snapshot / release attachment).
    for pattern, dest in (("*grok-workspace.zip", root / "_workspace_source"),
                          ("Metatron_Full_System.zip", root / "_metatron_source")):
        for outer in root.rglob(pattern):
            if not _extract_zip(outer, dest):
                continue
            # The grok workspace nests the release under attachments/.
            for nested in list(dest.rglob("Metatron_Full_System.zip")):
                _extract_zip(nested, dest / "release")
            found = list(dest.rglob("metatron_v2.py"))
            if found:
                return _load_from_path(found[0])

    raise FileNotFoundError(
        "metatron implementation not found (vendored package, metatron_v2.py, "
        "or full-system zip)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--scale", default=os.getenv("METATRON_SCALE", "ultra"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--text", default=os.getenv("METATRON_TEXT", ""))
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--parent", default="")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    mod = load_metatron(root)

    ck = Path(args.checkpoint or
              (os.getenv("METATRON_CHECKPOINT_DIR", "") + "/model.pkl"))
    ck.parent.mkdir(parents=True, exist_ok=True)
    parent = Path(args.parent) if args.parent else root / "daycare_state" / "champion" / "model.pkl"

    if parent.is_file():
        with open(parent, "rb") as f:
            model = pickle.load(f)
        parent_desc = str(parent)
    else:
        scale = getattr(mod, "SCALES", {}).get(args.scale)
        if scale is None:
            raise SystemExit(f"unknown scale {args.scale!r}; "
                             f"available: {sorted(getattr(mod, 'SCALES', {}))}")
        model = mod.MetatronV2(scale)
        parent_desc = "fresh-init"

    # Conservative optimizer settings: the daycare gates any later relaxation
    # behind a measured benchmark improvement.
    safe_lr = float(os.getenv("METATRON_SAFE_LR", "0.0001"))
    safe_clip = float(os.getenv("METATRON_GRAD_CLIP", "0.5"))
    model.cfg.learning_rate = min(float(model.cfg.learning_rate), safe_lr)
    model.cfg.grad_clip = min(float(model.cfg.grad_clip), safe_clip)

    text = (Path(args.text).read_text(encoding="utf-8")
            if args.text and Path(args.text).is_file() else mod.DEFAULT_TEXT)
    epochs = max(1, args.epochs if args.epochs is not None
                 else int(os.getenv("METATRON_BUDGET", "25")) // 25)

    mod.train(model, text, epochs=epochs,
              out_dir=str(ck.parent / "numpy_diagnostics"))

    tmp = ck.with_suffix(ck.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, ck)
    print(json.dumps({"status": "ok", "checkpoint": str(ck), "scale": args.scale,
                      "epochs": epochs, "parent": parent_desc}))


if __name__ == "__main__":
    main()
