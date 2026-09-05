# Metatron Pokémon Daycare

Metatron's **Daycare** is a persistent training/evolution controller. The Pokémon framing is the product metaphor: the model has a species, level, XP, moves/capabilities, parents, candidates, and evolution events. The underlying process remains ordinary machine learning and software evaluation.

## The critical design rule

**The laptop is not the daycare. The remote worker is.** If the laptop is shut down, local compute stops. A queued job continues only when a remote worker/CI runner/cloud GPU is running. State and checkpoints therefore live on the worker's durable storage, not only in browser memory.

```text
                 METATRON POKÉMON
                       │
                 ┌─────▼─────┐
                 │  DAYCARE  │  durable queue + XP + ledger
                 └─────┬─────┘
                       │
          observe → hatch → train → test
                       │              │
                       └──────┬───────┘
                              ▼
                         BENCHMARK GATE
                         /           \
                    pass             fail
                     │                 │
                  promote          discard/keep
                     │
                 CHAMPION
                     │
                GGUF / Ollama
```

## Low-energy policy

1. **Event-driven, not busy-looping.** A worker sleeps between jobs and can instead be invoked by a scheduler.
2. **Cheap mutation first.** Curriculum/data changes and LoRA/adapter experiments precede architecture rewrites.
3. **Adaptive budgets.** The next experiment starts small; compute increases only after measured gains.
4. **Checkpoint every experiment.** A crash or shutdown resumes from durable state rather than restarting training.
5. **Benchmark before promotion.** A candidate cannot replace the champion merely because training loss improved.
6. **Deduplicate work.** Candidate IDs and parent lineage make experiments reproducible.
7. **No fake learning.** The orchestrator records real trainer/evaluator output; it never claims progress when no trainer ran.

## Trainer contract

Set `METATRON_TRAINER` to a command that reads:

- `METATRON_PARENT`
- `METATRON_CANDIDATE`
- `METATRON_KIND`
- `METATRON_BUDGET`
- `METATRON_CHECKPOINT_DIR`

The trainer should write resumable checkpoints under `METATRON_CHECKPOINT_DIR`.

Set `METATRON_EVALUATOR` to a command that prints one final JSON line:

```json
{"score": 0.731, "metrics": {"loss": 1.92, "coding": 0.81, "reasoning": 0.64}}
```

## Example

```bash
python daycare/metatron_daycare.py --once \
  --trainer "python train.py" \
  --evaluator "python evaluate.py"
```

For an always-on remote worker:

```bash
python daycare/metatron_daycare.py \
  --trainer "python train.py" \
  --evaluator "python evaluate.py"
```

For a safe wiring test with no training:

```bash
python daycare/metatron_daycare.py --once --dry-run
```

## External learning sources

Ingestion should produce normalized, provenance-aware records before training:

- GitHub: repository, commit SHA, path, license, retrieval time
- Hugging Face: dataset/model/revision, license, retrieval time
- LLM teachers: provider/model/version, prompt template, timestamp
- Moltbook or other feeds: URL/post ID, retrieval time, provenance and usage policy

Do not dump external material directly into the weights. First deduplicate, filter, attach provenance, and run a curriculum/quality gate.

## Evolution policy

The safest recursive loop is:

**observe → propose → implement/train → test → benchmark → review → accept/reject → checkpoint**

The Pokémon metaphor does not imply autonomous agency or consciousness. "Evolution" means a measurable change in model weights, adapters, data curriculum, tool configuration, or code that survives the benchmark gate.
