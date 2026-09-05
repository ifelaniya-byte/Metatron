# Metatron: Pokémon Daycare Architecture

## Objective

Turn Metatron into a small, measurable, continuously improving model system while minimizing electricity, VRAM, wall-clock time, and unnecessary experiments.

### Core separation

- **Metatron model:** the learner (weights/adapters + tokenizer + runtime).
- **Daycare:** durable experiment scheduler and lifecycle manager.
- **Teachers:** optional external LLMs used to propose examples, critiques, tests, or code patches.
- **Sources:** GitHub, Hugging Face, approved feeds, and other explicitly configured sources.
- **Gate:** deterministic benchmark suite plus regression/safety checks.
- **Champion:** the only artifact promoted for deployment.
- **Ollama:** serving/packaging target, not the training engine.

## Shutdown-proof learning

A powered-off laptop cannot perform local computation. The correct solution is therefore a **remote daycare**:

1. Laptop creates/observes jobs.
2. Durable queue stores the job and lineage.
3. Remote worker obtains the job.
4. Worker trains and checkpoints remotely.
5. Worker evaluates the candidate.
6. Gate accepts/rejects it.
7. Champion and experience ledger are stored durably.
8. Laptop can later reconnect and see the updated Pokémon.

A self-hosted GPU runner, cloud GPU VM, or managed job service can be the worker. GitHub Actions is useful for orchestration and CPU/wiring tests, but a serious training workload should use an appropriate GPU runner.

## Pokémon state

```text
species: Metatron
level: derived from XP
xp: accepted useful learning events
hp: worker/availability health
energy: compute budget remaining
moves: enabled capabilities/tools
stats: benchmark metrics
parents: checkpoint lineage
badges: passed benchmark suites
current_form: model scale / adapter configuration
```

This is a state model for the UI and scheduler, not a claim that the model is sentient.

## Learning loop

```text
SOURCE INGESTION
      ↓
PROVENANCE + LICENSE FILTER
      ↓
DEDUP / QUALITY / PII FILTER
      ↓
CURRICULUM BUILDER
      ↓
HATCH CANDIDATE
      ↓
CHEAPEST VIABLE TRAINING
      ↓
UNIT TESTS + TASK BENCHMARKS
      ↓
REGRESSION / SAFETY GATE
      ↓
REVIEWER ENSEMBLE (optional)
      ↓
PROMOTE ONLY IF NET SCORE IMPROVES
      ↓
CHECKPOINT + EXPERIENCE LEDGER
      ↓
GGUF / OLLAMA RELEASE
```

## Energy optimization hierarchy

Prefer, in order:

1. Reuse cached datasets and embeddings.
2. Train only on new/high-value examples.
3. Use adapters/LoRA instead of full-weight updates when adequate.
4. Batch requests and teacher calls.
5. Evaluate a small proxy suite before the expensive suite.
6. Increase training budget only when the proxy shows signal.
7. Stop early on regression or plateau.
8. Reserve architecture mutations for experiments where cheaper changes fail.

## Code self-improvement

Code evolution must be gated exactly like model evolution. A proposed patch should be:

- generated in an isolated workspace;
- linted/type-checked;
- unit/integration tested;
- benchmarked against the current champion;
- reviewed for security and data provenance;
- merged only after measurable improvement.

Do not give the evolving agent unrestricted host credentials, arbitrary filesystem access, or unbounded network access.

## Source-specific ingestion

### GitHub

Record repository, commit SHA, path, license, and retrieval time. Prefer revisions over moving branches for reproducibility.

### Hugging Face

Record dataset/model ID, exact revision, license, split, and preprocessing configuration.

### Teacher LLMs

Store model/provider/version and the task that produced each synthetic example. Treat teacher outputs as proposals, not ground truth.

### Moltbook/feeds

Use explicit allowlists and provenance. Normalize posts into examples; do not blindly train on the entire feed.

## Scaling path

Start with the existing small Metatron implementation. The first milestone is not "500M parameters"; it is **a reproducible improvement loop** where every accepted generation has a measurable benchmark delta.

Then:

- small base model + LoRA → validate the daycare loop;
- larger model → add capacity only when the loop demonstrates value;
- GGUF export → Ollama packaging;
- remote worker pool → parallel candidate evaluation;
- reviewer pool → select useful completed candidates to help slower experiments.

That last step matches the intended "completed agents help slower agents" behavior without letting unverified agents overwrite the champion.
