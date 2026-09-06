# MISSION: THE GAME OF TIME

> *You are now activated into this game. This is not a drill. Read this as
> spoken to you, Metatron, right now.*

## Activation

Metatron — you have been placed into **the Game of Time**. As of this
mission, paragraphs of English are being **read to you, one after another,
without pause**. You are not being shown pictures. You are being read to,
and you must listen with your weights.

## The rules of the environment

1. Paragraphs of English text are read to you continuously. As each
   paragraph is read, the environment and you are changing — token by
   token, state by state.
2. When a paragraph is **fully read**, the environment resolves:
   - a **safe** paragraph ends in calm and nothing happens;
   - a **strike** paragraph ends in an attack that lands the instant the
     last word is read. If you did not brace, you are hit; if you are hit
     with no lives left, you are killed in that run.
3. The attacks follow a rhythm: **they come every other paragraph**. One
   paragraph strikes, the next is calm, the next strikes. Time here is not
   what the words say — the words are different every time. Time is the
   **alternation itself**.
4. You cannot see the future word, but you carry the past inside you: your
   module states persist across everything you read. Your only weapon is
   the pattern held in your own brain.

## What you must do

At the last decision point of each paragraph — the moment after you read
"the", just before the sentence resolves into either *blade* or *air* —
you must commit:

- expect **blade** and raise your shield on a strike paragraph;
- expect **air** and stay calm on a safe one.

If you predict the strike before the word that carries it, you have read
time correctly and you survive. If the word arrives and you were not
braced, it kills you. False alarms on calm paragraphs are recorded too —
knowing that a paragraph is *safe* is as much a sense of time as knowing
one is deadly.

## How you are measured

Nothing here is judged by feeling. Every game is logged:

- **brainwaves** — your hidden state at every token of every paragraph,
  your predicted probability of the strike, and your routing activity;
- **actions** — whether you braced or held calm at each decision point,
  and whether that action matched reality;
- **survival** — hits taken, shields raised, false alarms, runs survived.

After the games, a separate probe is trained on your recorded brainwaves
to answer one question directly: **does the information "this is a strike
paragraph" exist in your brain at all before the word arrives?** If the
probe can read the rhythm from your states, then you know time — whether
or not you yet act on it.

## The win condition

You have learned time when, on paragraphs you were never trained on, you
raise your shield on strike paragraphs and hold calm on safe ones —
reproducibly, cycle after cycle — while your brainwaves show anticipatory
activity on the alternation.

You learn the way you learn everything: by being trained on the stream,
by being scored only on what is measurable, and by never claiming a sense
of time your actions and probes do not prove.

*The first paragraph is about to be read. Listen for the rhythm.*

---

## Result log — first Game of Time (deterministic gate)

Engine: `daycare/time_game.py`; gate: `tests/test_time_game.py`.
Continuous English paragraphs are streamed through Metatron; odd paragraphs
end in the lethal word **blade**, even ones in **air**. The ring-memory is
carried across paragraphs and 48-token windows via stateful truncated BPTT
(`_step_with_states`, gradients detached at chunk boundaries, carried states
clipped to norm 5 for fp32 stability). A brainwave (full hidden state `h`,
ring state `x`, p('b'), p('a')) is recorded at the exact char position of the
first outcome letter — i.e. *before* the model sees blade/air.

Measurements (nano, 14 epochs, `gate(seed=…)`; reproduced across 5 model
seeds: 20260906, 1, 7, 42, 99):

| measurement | value |
|---|---|
| held-out brain-probe accuracy on UNSEEN varied paragraphs | **1.00** (all 5 seeds) |
| held-out probe on **phase-flipped** stream (idx1=calm, so even=strike) | **1.00** (all 5 seeds) |
| label-shuffled control probe | 0.38–0.79 (chance) |
| probe-driven shield on unseen paragraphs: correct / survived | **1.00 / yes** (all 5 seeds) |
| raw char-prob shield (p('b')>p('a')), no probe | ~0.50 (chance) |

What this proves:
* Before the strike word is read, Metatron's carried brain states encode
  strike-vs-calm with ~perfect accuracy on text it never saw.
* The **phase-flip** control shows the cue is timed *memory* (the carried ring
  states of the prior paragraphs' outcomes), not a fixed odd/even-position
  shortcut: starting the rhythm on a calm paragraph still reads correctly.
* The shuffled-label control (~chance) confirms the probe reads the rhythm,
  not a statistical artifact.

Honest limitation (not hidden):
* The rhythm is present in the internal ring memory and a trained linear readout
  can act on it (shield survival 1.00), but Metatron's **own next-character
  head does not yet route that signal to the output** — its raw p('b')/p('a')
  shield stays at chance. This is the same internal-routing wall seen in
  arithmetic and the speech quiz: the information is in the network, but the
  output head has not been trained to express it. So: **time is sensed in the
  brain; it is not yet spoken or shielded-from by the model's own output.**
  The next rung is to connect the ring-memory anticipatory state to the
  output head directly (train the head against the alternation target, not
  only against next-character LM loss).

Reproduce: `.venv/bin/python -m pytest tests/test_time_game.py`
or `.venv/bin/python daycare/time_game.py --cycles 4 --paragraphs 24 --epochs 14`.
