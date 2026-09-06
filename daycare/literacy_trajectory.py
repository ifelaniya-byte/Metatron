#!/usr/bin/env python3
"""Bounded trajectory probe: how many epochs until Metatron stops babbling?

Trains nano from scratch on the literacy curriculum and reports word
validity every few epochs. Used to choose the daycare loop budget.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metatron import MetatronV2, SCALES, train  # noqa: E402
from daycare.literacy import literacy_metrics, passed_literacy  # noqa: E402

text = Path("daycare_state/curriculum/literate.txt").read_text(encoding="utf-8")


def main():
    scale = sys.argv[1] if len(sys.argv) > 1 else "nano"
    epochs_per_block = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    m = MetatronV2(SCALES[scale])
    m.cfg.learning_rate = min(float(m.cfg.learning_rate), 0.002)
    m.cfg.grad_clip = min(float(m.cfg.grad_clip), 0.5)
    for block in range(blocks):
        with contextlib.redirect_stdout(io.StringIO()):
            train(m, text, epochs=epochs_per_block,
                  out_dir="/tmp/lit_ckpt", verbose=False)
        mt = literacy_metrics(m)
        done = "LITERATE" if passed_literacy(mt) else ""
        print(f"after {(block+1)*epochs_per_block:2d} epochs: "
              f"loss {mt['literacy_loss']:.3f} "
              f"ppl {mt['literacy_perplexity']:6.1f}  "
              f"word_validity {mt['word_validity']:5.1%}  "
              f"unique {mt['word_unique']:4.0%}  "
              f"words {int(mt['word_count']):3d}  {done}",
              flush=True)
        if passed_literacy(mt):
            break


if __name__ == "__main__":
    main()
