# Reference archive

These files are the original Metatron build artifacts delivered with the
project (500M-parameter scaling templates, earlier architecture sketches,
training guides). They are kept for provenance and as scaling references —
they are **not** the live implementation.

The live, maintained code is the vendored package at the repository root:

- [`metatron/metatron_v2.py`](../../metatron/metatron_v2.py) — the
  learnable Flower-of-Life model with **real backpropagation-through-time**
  (all parameters receive gradients; verified by finite differences in
  `tests/test_metatron.py`), Adam updates, capability verification, and
  pickle/npz persistence.

Known limitation of the archived `metatron_v2.py` in this folder: its
training loop only accumulated gradients into the output head
(`head.backward` inside `loss`), while the geometric attention modules'
`backward_and_step` was a no-op (`_dW` was never populated) and the embedding
only received weight decay. The body of that network therefore never learned.
The root package fixes this end-to-end.
