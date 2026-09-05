# Metatron Pokémon Daycare

Metatron is treated as a **Pokémon**: it has a species, generation, XP, level, lineage, moves/capabilities, parents, candidates, and a champion. The machinery underneath is ordinary ML training, benchmarking, and software evolution.

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
    ├─ real training
    ├─ benchmark + review gate
    └─ champion checkpoint
             │
             ▼
      Ollama-compatible bridge
```

## Energy policy

1. Event-driven scheduling; no busy spinning.
2. Cheap curriculum experiments before expensive architecture changes.
3. Conservative learning rate and adaptive budgets.
4. Full model-object checkpoints so internal NumPy weights and optimizer state survive process restarts.
5. Bounded-context evaluation to match the current Metatron recurrent geometry.
6. Promotion only on a measurable benchmark gain.
7. Cache/deduplicate work and keep provenance for external material.

## Real trainer

`daycare/train_real.py` discovers `metatron_v2.py` inside the checked-out workspace/`Metatron_Full_System.zip`, starts from the current champion when available, trains the actual NumPy Metatron implementation, and writes `model.pkl` under `METATRON_CHECKPOINT_DIR`.

The adapter intentionally clamps the initial learning rate to `1e-4` and gradient clip to `0.5` because the supplied release becomes numerically unstable at its original learning rate. Those values are configurable with `METATRON_SAFE_LR` and `METATRON_GRAD_CLIP` and should only be relaxed after the benchmark proves stability.

## Benchmark gate

`daycare/evaluate_real.py` runs capability checks on a copy (so evaluation cannot mutate the candidate) and measures loss on the same bounded context windows used by training. Candidates producing NaN/Inf are rejected.

## Remote scheduler

`.github/workflows/metatron-daycare.yml` schedules a daycare cycle every six hours and can also be manually dispatched. Set `METATRON_RUNNER` to a persistent self-hosted GPU label for serious training; the GitHub-hosted CPU runner is intended for wiring/smoke tests. The champion is restored/saved with Actions cache and the small experience ledger is committed back to the branch.

## Ollama

The custom NumPy architecture is **not** a native Ollama/GGUF architecture, so the system does not falsely claim that it can be converted to GGUF today. Instead, `daycare/ollama_bridge.py` exposes the real persisted champion through Ollama's HTTP API shape.

Run on the remote worker:

```bash
python daycare/ollama_bridge.py --checkpoint daycare_state/champion/model.pkl --port 11435
```

Then point an Ollama client at the bridge endpoint. Native `ollama create`/GGUF packaging is a separate compatibility project requiring a supported Ollama backend or a custom runtime implementation.

## External learning

`daycare/source_ingest.py` accepts explicit URLs, caps downloads, hashes content, and records provenance. `daycare/teacher_ensemble.py` optionally rotates across OpenAI-compatible teacher endpoints. Review licensing/usage rights before putting external material into training.

Recommended pipeline:

**ingest → deduplicate → provenance/license gate → teacher/curriculum generation → cheap training → benchmark → reviewer → promote/reject → checkpoint**

No component reports learning unless the trainer actually ran and the benchmark produced a valid result.
