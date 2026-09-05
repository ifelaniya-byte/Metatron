# Metatron

A small, measurable, continuously improving language-model system, treated
as a **Pokémon** that levels up in a "daycare": it hatches candidates, trains
them with real gradients, benchmarks them, and only promotes a champion when
a held-out score improves.

Metatron's learner is a pure-NumPy **Flower-of-Life geometric modular
network**: a token walks a ring of modules, each a Pre-Norm residual block of
multi-head geometric message passing (persistent module states serve as KV
memory, with a learned router over the fixed topology) plus a SwiGLU
feed-forward block. It competes with frontier models not by pretending to
have frontier capacity, but through the complete system — persistent memory,
tooling, gates, and cheap, reproducible improvement loops (see
[FRONTIER_OUTPACE_PLAN.md](FRONTIER_OUTPACE_PLAN.md) and
[METATRON_DAYCARE_ARCHITECTURE.md](METATRON_DAYCARE_ARCHITECTURE.md)).

## Layout

| Path | What it is |
| --- | --- |
| `metatron/metatron_v2.py` | The live learner: real backpropagation-through-time, Adam, capability verification, pickle/npz persistence. |
| `daycare/` | Shutdown-proof orchestrator: trainer/evaluator adapters, durable state + lineage, Ollama-compatible bridge, source ingestion, teacher ensemble, frontier arena. |
| `tests/` | Self-generated executable gate: finite-difference gradient checks, all-parameter gradient reach, overfit tests, end-to-end daycare promotion. |
| `.github/workflows-available/metatron-daycare.yml` | Active-by-default schedule: tests first, then one hatch → train → evaluate → promote cycle. Move into `.github/workflows/` after merge (the branch token can't publish workflow files). |
| `docs/reference/` | Archived scaling templates and the original release artifacts (provenance, not live code). |

## Quick start

```bash
pip install -r requirements.txt

# Train the smallest scale directly (CPU, seconds per epoch)
python -m metatron.metatron_v2 --scale nano --epochs 10

# The non-negotiable gate: finite-difference gradient checks + capabilities
# + end-to-end daycare train/evaluate/promote
python -m pytest tests/ -q

# One full daycare cycle locally (state lands in daycare_state/)
python daycare/metatron_daycare.py --once \
  --trainer "python daycare/train_real.py --root . --epochs 3" \
  --evaluator "python daycare/evaluate_real.py --root ."

# Serve the promoted champion through an Ollama-compatible API
python daycare/ollama_bridge.py --checkpoint daycare_state/champion/model.pkl --port 11435
```

## The gate

No "Metatron is smarter" claim is permitted from training loss, confidence,
or teacher praise. Accepted evidence is only:

1. the finite-difference / capability test gate (`tests/`);
2. a reproducible benchmark delta on held-out bounded-context windows
   (`daycare/evaluate_real.py` emits `score` + `metrics` JSON);
3. resource-efficiency measurements logged in the experience ledger.

Teacher models (frontier LLMs) generate tasks, critiques, and regression
tests; their output is treated as proposals until an executable check
passes.
