# Metatron Pokémon Daycare

Metatron is treated as a **Pokémon**: it has a species, generation, XP, level, lineage, moves/capabilities, parents, candidates, and a champion. The machinery underneath is ordinary ML training, benchmarking, and software evolution.

## Status: active

The daycare is **active by default**: `.github/workflows/metatron-daycare.yml`
runs on a six-hour schedule and on manual dispatch, executes the
self-generated test gate first, then trains a real candidate, benchmarks it,
and promotes it only on a measured gain. GitHub-hosted CPU runners perform a
fast `nano` smoke cycle; point the repository variable `METATRON_RUNNER` at a
self-hosted GPU label and set `METATRON_SCALE=ultra` (or larger) for real
training. Set `METATRON_DISABLE_DAYCARE=true` to pause the schedule.

## Shutdown-proof daycare

**The laptop is only the trainer's controller. The remote worker is the daycare.** Turning the laptop off stops local compute, but it does not stop a remote GPU/runner. The champion checkpoint is cached on the remote worker and the small Pokémon experience state is persisted separately.

```text
LAPTOP / UI
    │ observe / issue commands
    ▼
REMOTE POKÉMON DAYCARE
    │
    ├─ durable state + lineage
    ├─ cheap experiment selection
    ├─ real training (real backpropagation)
    ├─ benchmark + capability gate
    └─ champion checkpoint
             │
             ▼
      Ollama-compatible bridge
```

## The learner

`metatron/metatron_v2.py` is the live implementation: a pure-NumPy
Flower-of-Life geometry where a token walks a ring of modules, each a
Pre-Norm residual block of multi-head geometric message passing (persistent
module-ring states act as KV memory, with a learnable router on top of the
fixed topology) plus a SwiGLU feed-forward block. Training uses **real
backpropagation-through-time** into every parameter (embeddings, position
table, Q/K/V/output/router projections, SwiGLU, norm scales, and the output
head), global-norm gradient clipping, and Adam with weight decay. The
gradient path is verified against central finite differences in
`tests/test_metatron.py` — no component reports learning without an
executable check.

Run it directly:

```bash
pip install -r requirements.txt
python -m metatron.metatron_v2 --scale ultra --epochs 20   # train
python -m metatron.metatron_v2 --verify                    # capability gate
python -m pytest tests/ -q                                 # full test suite
```

## Energy policy

1. Event-driven scheduling; no busy spinning.
2. Cheap curriculum experiments before expensive architecture changes.
3. Conservative learning rate and adaptive budgets.
4. Full model-object checkpoints so NumPy weights **and Adam moments** survive process restarts (pickle; `model.pkl`).
5. Bounded-context evaluation to match the recurrent geometry.
6. Promotion only on a measurable benchmark gain.
7. Cache/deduplicate work and keep provenance for external material.

## Real trainer

`daycare/train_real.py` imports the vendored `metatron` package (falling back
to any `metatron_v2.py` in the workspace, then to the archived full-system
zip), starts from the current champion when available, trains the actual
NumPy Metatron implementation, and writes `model.pkl` under
`METATRON_CHECKPOINT_DIR`.

The adapter clamps the initial learning rate to `METATRON_SAFE_LR`
(default `8e-4`) and gradient clip to `METATRON_GRAD_CLIP` (default `0.5`)
for stable hill-climbing; relax them only after the benchmark proves
stability.

## Benchmark gate

`daycare/evaluate_real.py` runs the capability suite on a **deep copy** (so
evaluation cannot mutate the candidate), then measures next-token loss over
the same bounded-context windows used by training. Candidates producing
NaN/Inf or failing any capability are rejected (`score = -inf`).

## Remote scheduler

`.github/workflows/metatron-daycare.yml`:

1. **tests job** — numerical correctness + capability tests on every run
   (this is the non-negotiable gate: no promotion without reproducible
   evidence);
2. **daycare job** — one hatch → train → evaluate → promote/reject cycle;
3. champion restored/saved via Actions cache; the small experience ledger
   (`state.json`, `experience.jsonl`, `queue.jsonl`, `pokemon.json`) is
   committed back to the branch; full diagnostics are uploaded as artifacts.

## Ollama-compatible bridge

The custom NumPy architecture is **not** a native Ollama/GGUF architecture,
so the system does not falsely claim a GGUF conversion. Instead
`daycare/ollama_bridge.py` serves the persisted champion through Ollama's
HTTP API shape:

```bash
python daycare/ollama_bridge.py --checkpoint daycare_state/champion/model.pkl --port 11435
curl -X POST http://localhost:11435/api/generate -d '{"prompt":"the flower of life"}'
```

Native `ollama create` / GGUF packaging is a separate compatibility project
requiring a supported Ollama backend or a custom runtime.

## External learning

`daycare/source_ingest.py` accepts explicit URLs, caps downloads, hashes
content, and records provenance. `daycare/teacher_ensemble.py` optionally
rotates across OpenAI-compatible teacher endpoints; `daycare/frontier_arena.py`
asks frontier models to act as adversarial teachers/judges. Teacher output
is **never** accepted as ground truth — only as proposals that must pass the
executable benchmark gate. Review licensing/usage rights before putting
external material into training.

Recommended pipeline:

**ingest → deduplicate → provenance/license gate → teacher/curriculum generation → cheap training → benchmark → reviewer → promote/reject → checkpoint**

No component reports learning unless the trainer actually ran and the benchmark produced a valid result.
