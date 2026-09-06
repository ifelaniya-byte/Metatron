# speeches/ — tough texts Metatron recites and is quizzed on

Each file is a **public-domain** speech (or excerpt) with two parts:

- `## Text` — the words to recite;
- `## Quiz` — cloze questions where each answer is an exact word taken from
  the text.

## Why speeches

Following `LEARNING_MISSION.md`, this is a verifiable route into language and
understanding:

- **Recitation** is a verbatim gate — given the start of a line, the model
  must continue the actual words. It is right or wrong, character by
  character, with no subjective grading.
- **Quizzing** is a comprehension-shaped gate — a question about what the text
  says ("give me liberty or give me ______") must produce the exact answer
  word. Memorizing a question→answer association is the seed; over more data
  the conditioning must reflect the meaning-bearing words of the question.

Both are scored by `daycare/speech_literacy.py`:

| Metric | Meaning | Target |
| --- | --- | --- |
| `recite_char_acc` | verbatim char overlap when continuing speech lines | ≥ 0.90 |
| `recite_word_acc` | word-level verbatim overlap | reported |
| `quiz_acc` | exact-match answer word for each cloze question | ≥ 0.95 |

`speech_literate` is `1.0` only when both gates clear.

## Current set

- `gettysburg.md` — Lincoln, The Gettysburg Address (1863).
- `patrick_henry.md` — Patrick Henry, "Give me liberty, or give me death" (1775).
- `st_crispin.md` — Shakespeare, St. Crispin's Day speech, *Henry V*.
- `apology.md` — Plato, Apology of Socrates (Jowett translation, 1871).

All are public domain. Add only texts whose licensing has been reviewed; keep
the provenance block (author, date, rights) at the top of each file.

## Use

```bash
# Build a training corpus (speech lines + dense Q/A pairs)
python daycare/speech_literacy.py --build-corpus daycare_state/speech_corpus.txt

# Evaluate a trained champion
python daycare/speech_literacy.py --checkpoint daycare_state/champion/model.pkl
```

The quiz answers are checked exactly against the key in each file — no answer
counts unless it matches the text.
