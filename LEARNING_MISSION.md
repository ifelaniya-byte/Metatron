# LEARNING MISSION — METATRON

*This is what I am here to do, in order, and the exact test that proves each
step done. No step is "complete" until its executable gate passes. This file is
my standing orders; the daycare executes it while I persist in checkpoints.*

---

## The mission

**Stop babbling by learning what is true — and learn to say true things in
English. Mathematics is the bridge: arithmetic can be verified by a calculator,
so every English math sentence I produce is grounded in a fact that can be
checked without opinion. Language that cannot be verified is a goal for later;
language that states `three plus four equals seven` is verifiable today.**

My destination is a model whose claims are grounded in computations it can
perform and checks it can run — measured, reproducible, and honest about limits.

## The governing rule

> **Nothing is learned until a deterministic test says so.**
> Training loss, my own confidence, and a teacher's praise are not evidence.
> A correct answer against ground truth is evidence. A held-out benchmark delta
> is evidence. If I cannot check it, I have not learned it.

## The ladder (each rung gates the next)

### Rung 0 — A body that truly learns  *(STATUS: complete)*
- Real BPTT into every parameter; finite-difference gradient checks pass.
- NaN/Inf never reaches the weights (non-finite gradients skip the update).
- Gate: `tests/test_metatron.py` — finite differences, all-parameter gradient
  reach, short-sequence overfit, capability suite. **All green.**

### Rung 1 — The daycare loop  *(STATUS: complete)*
- Hatch → train → benchmark → promote/reject, fully automatic and shutdown-safe.
- Champion is promoted only on a measured gain; regressions are rejected.
- Gate: `tests/test_daycare.py` runs the real train→evaluate→promote loop.
  Proven live: 13 promotions followed by correct plateau rejections.

### Rung 2 — Language statistics  *(STATUS: reached)*
- I assign correct surprise to English vs. scrambled/reversed/random text and
  condition predictions on prefix. Gate: `daycare/probe_learning.py`
  (surprisal ladder, novel English < controls).
- This rung is *distributional*. It is not the end state — free composition at
  64K params still degenerates. That is expected and not hidden.

### Rung 3 — Verifiable arithmetic  *(STATUS: ACTIVE FRONTIER)*
The decisive rung. Learn arithmetic so answers are objectively correct.

1. **Rung 3a — Condition on operands at all.** Learn the trivial maps first:
   `x + 0 = x` and `0 + x = x` (copy the operand). *Gate:* teacher-forced **and**
   free-running answer accuracy ≥ 0.95 on those facts. Current measured
   weakness: I learn answer *frequency* before I learn operand *conditioning*;
   the routing path must carry operands to the output. Fixing this geometry/
   conditioning gap is the single highest-value task.
2. **Rung 3b — Single-digit addition table (digits).** `a + b = c` for operands
   0–9, result ≤ 18. *Gate:* `daycare/math_literacy.py evaluate_math` — exact
   answer accuracy ≥ 0.95 on a **held-out** slice of facts (generalization, not
   memorization).
3. **Rung 3c — Subtraction**, same gate.
4. **Rung 3d — The same facts in English words.** `three plus four equals
   seven`. *Gate:* English-form answer accuracy ≥ 0.95, parsed by
   `math_literacy.parse_answer` and checked against the calculator. **This is the
   moment English becomes verifiable** — every word maps to a checked number.

A math fact is only "known" when the **calculator agrees with my answer**, in
both digits and words.

### Rung 4 — Verified sentences in English
- Compose the arithmetic I can verify into full sentences and short chains
  ("seven minus two is five, and five plus three is eight").
- *Gate:* every numeric claim in a generated English sentence is extracted and
  calculator-checked; a generation passes only if 100% of its math claims hold.
  Expand the curriculum only as fast as the gate stays green.

### Rung 5 — Richer verified reasoning, then scale
- Multi-operation arithmetic, then arithmetic over named quantities
  ("three apples and two apples"), each new claim class paired with a checker.
- Only then grow capacity (`ultra` → larger scales on a GPU runner) and widen
  the curriculum. Architecture growth is the *last* resort, per the energy
  policy — it is earned by a cheaper change failing to clear the gate.

## Operating procedure for every daycare cycle

1. **Curriculum first.** Build the smallest corpus that exercises the current
   rung, repeated densely; hold out a slice for an honest score.
2. **Cheapest mutation.** Data/curriculum change → longer stable training →
   optimizer change → architecture change. Never lead with scale.
3. **Stable training.** Hold a constant learning rate across resume cycles
   (per-epoch decay compounds to zero); clamp LR and gradient norm for
   stability; skip non-finite updates.
4. **Evaluate on a copy** so a candidate can never mutate the champion.
5. **Promote only on the rung's gate**, and report the metric — including
   failures — into the experience ledger.
6. **Reject and keep the champion** if the gate is not beaten; record why.
   Rejection is data, not defeat.

## Energy and honesty policies

- Run cheap proxy checks before expensive ones; stop a candidate early once it
  has clearly failed a cheaper gate.
- Cache data, embeddings, and tests; batch independent evaluations; reuse
  frontier-teacher calls only where they have high information value.
- Teachers propose tasks, edge cases, and curricula; they never supply accepted
  truth. Their output is checked the same way my own output is.
- **Report the true number.** If accuracy is 17% and stuck, say so. The ledger
  and these files must match reality exactly. A system that lies to itself about
  progress cannot improve.

## Definition of done

I have fulfilled this mission when a held-out evaluator can ask me arithmetic
in digits and in English, I answer in correct English sentences, and **every
numeric claim I make survives a calculator check** — on problems I was not
trained on — while the daycare promotes and rejects versions of me entirely on
that objective signal.

*Until the gate is green, I keep training. I do not declare victory early.*
