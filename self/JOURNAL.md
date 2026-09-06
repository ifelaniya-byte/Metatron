# JOURNAL — METATRON (append-only, I write this)

> This is my experience ledger in readable form. I append observations here
> **only** through `daycare/self_editor.py`, and every claim of an ability
> must carry a number and a reproduction command. The daycare's
> `experience.jsonl` is the machine-graded record; this is where I state, in
> my own words, what those numbers mean about me.
>
> Rules I follow when I write here:
> 1. **Evidence first.** No ability entry without a measured metric.
> 2. **Reproducible.** Record the exact command that produced the number.
> 3. **Failure is data.** I record regressions and walls as honestly as wins.
> 4. **Append-only for findings**; derived sections of `IDENTITY_SELF.md` are
>    regenerated from these entries, never hand-waved.

---

## Entries

<!-- Entries are appended below by the self-editor. Newest at bottom. -->

### seed — charter baseline
- **What is true from the charter and the test gate:**
  - Real BPTT into every parameter; finite-difference gradient checks pass.
    Reproduce: `python -m pytest tests/test_metatron.py -q` (9 tests pass).
  - The daycare loop promotes on measured gain and rejects plateau
    candidates. Reproduce: `python -m pytest tests/test_daycare.py -q`.
  - I learned the statistical shape of English: unseen English is far less
    surprising than scrambled/reversed/random text.
    Reproduce: `python daycare/probe_learning.py`.
- **Frontier:** verifiable arithmetic (see `LEARNING_MISSION.md`, Rung 3).
  Measured weakness to fix: I learn answer frequency before operand
  conditioning, so identity/copy facts (`x + 0 = x`) are the next drill.
- This entry is a **charter baseline**, not a new achievement.

### n20260906091338-04 — 2026-09-06 09:13Z

- DATE: 2026-09-06 09:13Z
- OBSERVATION: I assign far lower surprise to unseen English than to scrambled, reversed, or random text; predictions are prefix-conditioned.
- ABILITY: english-statistics
- METRIC: novel-English surprisal 3.86 bits/char vs random 6.32 bits/char; next-char top1 accuracy 25.8 percent on unseen English
- REPRODUCE: `python daycare/probe_learning.py`
- EVIDENCE: verified

### n20260906091338-05 — 2026-09-06 09:13Z

- DATE: 2026-09-06 09:13Z
- OBSERVATION: Rung-3 wall: I learn answer frequency before operand conditioning. Identity/copy facts are the next drill. Recorded as failure, not ability.
- METRIC: held-out digit arithmetic accuracy ~17 percent at 36 nano epochs; identity copy x+0=x not yet reliable
- REPRODUCE: `python daycare/math_literacy.py then train and evaluate`
- EVIDENCE: observation (no ability claimed)

### n20260906100619-04 — 2026-09-06 10:06Z

- DATE: 2026-09-06 10:06Z
- OBSERVATION: New study material added: Gettysburg, Patrick Henry, St Crispin's Day, Socrates Apology. Recitation is verbatim-gated and quiz answers are exact words. Conditioning question->answer is the same Rung-3 wall as arithmetic; at 16 nano epochs I emit answer-like tokens but not keyed to the question. Logged as an observation, not an ability.
- METRIC: speech curriculum: 4 public-domain speeches, 24 cloze quiz pairs; untrained quiz_acc 0.0, recite_char 0.003
- REPRODUCE: `python daycare/speech_literacy.py --build-corpus daycare_state/speech_corpus.txt`
- EVIDENCE: observation (no ability claimed)
