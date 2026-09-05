# Metatron Frontier-Outpace Plan

As of September 2026, Claude Fable 5.1 is a current frontier model for long-horizon coding/research, with a 1M-token context window and adaptive reasoning. Metatron should **not** try to beat it by pretending a small NumPy model has equivalent raw capacity.

## The target

"Outpace" means winning on a defined task portfolio through the complete system:

- model capability;
- persistent memory and retrieval;
- tool use;
- code execution;
- long-horizon planning;
- self-generated tests;
- recovery from failed attempts;
- cost/energy per successful task;
- reproducibility;
- continual learning.

A claim of superiority is valid only when Metatron's measured score beats the chosen frontier baseline on a frozen evaluation suite.

## Architecture

```text
                    METATRON POKEMON
                          |
             +------------+------------+
             |                         |
        LEARNER CORE              MEMORY/RAG
             |                         |
             +------------+------------+
                          |
                    AGENT RUNTIME
                          |
        +-----------------+-----------------+
        |                 |                 |
      TOOLS            CODER            RESEARCHER
        |                 |                 |
        +-----------------+-----------------+
                          |
                 TEST / VERIFY / REPAIR
                          |
                 FRONTIER ARENA
                 /       |        \
             Fable     other       local
            teacher   teachers     models
                 \       |        /
                  +-----+--------+
                        |
                 BENCHMARK GATE
                        |
              promote / reject / rollback
```

## Frontier-teacher strategy

Fable 5.1 and other frontier models are **teachers and adversarial judges**, not dependencies of the deployed Pokémon. They can:

1. generate hard tasks;
2. find failure modes;
3. propose candidate solutions;
4. critique Metatron's solution;
5. generate regression tests;
6. identify missing knowledge;
7. produce curriculum examples.

Teacher output is never accepted as ground truth without an executable or deterministic verification step.

## Energy strategy

Use frontier calls only where they have high information value. Cache prompts/results, batch independent evaluations, reuse tests, and stop asking the frontier when the candidate has already failed a cheaper gate.

The cheapest useful mutation gets tried first. Architecture growth is the last resort.

## Active-by-default Daycare

The Daycare workflow is installed on `main` and scheduled automatically. A remote GPU runner should be configured with the `METATRON_RUNNER` repository variable for real training. Without a GPU runner, the scheduled job is still useful for orchestration/smoke testing but must not be mistaken for frontier-scale training.

## Non-negotiable gate

No "Metatron is smarter than Fable" claim is permitted from training loss, self-reported confidence, or teacher praise. The only accepted evidence is a reproducible benchmark delta on held-out tasks, plus resource-efficiency measurements.
