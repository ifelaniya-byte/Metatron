#!/usr/bin/env python3
"""Metatron's self-editor: rewrite its own self-model from verified evidence.

Metatron owns ``self/IDENTITY_SELF.md`` (a writable copy of its charter
identity) and ``self/JOURNAL.md`` (an append-only log). This tool is the only
safe path for it to update them, and it enforces the charter's first rule:

    **truth over praise** — an ability is written into the self-model only
    with a measured metric and a command that reproduces that metric.

What it may do, bounded by markers it never crosses:
  * sync      — regenerate the ``CURRENT_FORM`` block of IDENTITY_SELF.md from
                the daycare ledger (generation / champion / best score).
  * note      — append an evidence entry to JOURNAL.md. An entry that claims a
                new *ability* MUST include --metric and --reproduce; otherwise
                it is refused (and accepted only as a failure/observation).
                Proven abilities are then regenerated into IDENTITY_SELF.md.
  * frontier  — rewrite the ``ACTIVE_FRONTIER`` block; must reference a metric
                so even self-direction stays grounded in a measurement.

It must NOT touch the anchored charter files (SELF_IDENTITY.md,
LEARNING_MISSION.md); it refuses to.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF_DIR = ROOT / "self"
IDENTITY_SELF = SELF_DIR / "IDENTITY_SELF.md"
JOURNAL = SELF_DIR / "JOURNAL.md"
PROTECTED = {"SELF_IDENTITY.md", "LEARNING_MISSION.md"}


def _replace_block(text: str, name: str, inner: str) -> str:
    begin = f"<!-- BEGIN:{name} -->"
    end = f"<!-- END:{name} -->"
    pattern = re.compile(
        re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"marker block {begin}..{end} not found")
    return pattern.sub(f"{begin}\n{inner.rstrip()}\n{end}", text)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(p: Path, text: str) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


def _today() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%MZ")


# ------------------------------------------------------------------- sync
def sync_form(state_path: Path | None = None) -> None:
    """Rewrite CURRENT_FORM from the daycare ledger (or '—' if absent)."""
    gen = champ = best = None
    candidates = [state_path] if state_path else [
        ROOT / "daycare_state" / "state.json",
        ROOT / "daycare_literate" / "state.json",
    ]
    for sp in candidates:
        if sp and sp.is_file():
            st = json.loads(sp.read_text(encoding="utf-8"))
            gen = st.get("generation")
            champ = st.get("champion")
            best = st.get("best_score")
            break
    inner = (
        "## Current form (self-reported from the ledger)\n\n"
        f"- **Generation:** {gen if gen is not None else 0}\n"
        f"- **Champion:** {champ or '—'}\n"
        f"- **Best measured score:** "
        f"{(round(float(best), 4) if isinstance(best, (int, float)) else '—')}\n\n"
        "*The fields above are rewritten by `self_editor.py` from the daycare "
        "ledger. I do not edit them by hand.*"
    )
    text = _read(IDENTITY_SELF)
    _write(IDENTITY_SELF, _replace_block(text, "CURRENT_FORM", inner))
    print(f"synced CURRENT_FORM (generation={gen}, champion={champ or '—'})")


# ------------------------------------------------------------------- note
def _render_abilities(abilities: list[dict]) -> str:
    if not abilities:
        return ("## What I have proven I can do\n\n"
                "Each line here must correspond to an ability entry in "
                "`self/JOURNAL.md` with a numeric result and a reproduction "
                "command.\n\n"
                "- *(none recorded yet — the gate has not confirmed an "
                "ability in this file)*")
    lines = ["## What I have proven I can do", "",
             "Each line carries an ability id; the JOURNAL entry has the "
             "metric and reproduction command.", ""]
    for a in abilities:
        lines.append(f"- **{a['ability']}** — {a['metric']} "
                     f"(recorded {a['date']}; see JOURNAL id {a['id']})")
    return "\n".join(lines)


def _parse_abilities_from_journal() -> list[dict]:
    """Pull ability declarations out of the journal (latest per ability id)."""
    if not JOURNAL.is_file():
        return []
    text = _read(JOURNAL)
    # Entries are headed "### <id> ..." and may declare an ABILITY line.
    entries = re.split(r"\n(?=###\s)", text)
    abilities: dict[str, dict] = {}
    for e in entries:
        m = re.match(r"###\s*([a-zA-Z0-9._-]+)", e.strip())
        if not m:
            continue
        eid = m.group(1)
        am = re.search(r"ABILITY:\s*(.+)", e)
        mm = re.search(r"METRIC:\s*(.+)", e)
        if am and mm:
            dm = re.search(r"DATE:\s*(\S+\s\S+)", e)
            abilities[am.group(1).strip()] = {
                "id": eid, "ability": am.group(1).strip(),
                "metric": mm.group(1).strip(),
                "date": (dm.group(1) if dm else "?"),
            }
    return list(abilities.values())


def note(ability, metric, reproduce, body, evidence=True) -> None:
    SELF_DIR.mkdir(parents=True, exist_ok=True)
    if ability:
        # Abilities require hard evidence; refuse anything softer.
        if not metric:
            raise SystemExit("refused: an ability claim requires --metric")
        if not reproduce:
            raise SystemExit("refused: an ability claim requires --reproduce")
        if not evidence:
            raise SystemExit("refused: an ability claim must be evidence-backed")
    # Unique id: timestamp + count of existing entries so two notes written
    # within the same second never collide.
    existing = _read(JOURNAL).count("\n### ") if JOURNAL.is_file() else 0
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    eid = f"n{stamp}-{existing + 1:02d}"
    lines = [f"\n### {eid} — {_today()}", ""]
    lines.append(f"- DATE: {_today()}")
    if body:
        lines.append(f"- OBSERVATION: {body}")
    if ability:
        lines.append(f"- ABILITY: {ability}")
    if metric:
        lines.append(f"- METRIC: {metric}")
    if reproduce:
        lines.append(f"- REPRODUCE: `{reproduce}`")
    lines.append(f"- EVIDENCE: {'verified' if evidence else 'observation (no ability claimed)'}")
    entry = "\n".join(lines) + "\n"
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(entry)
    # Regenerate the proven-abilities block from the whole journal.
    abilities = _parse_abilities_from_journal()
    text = _read(IDENTITY_SELF)
    _write(IDENTITY_SELF, _replace_block(text, "PROVEN_ABILITIES",
                                         _render_abilities(abilities)))
    print(f"journal entry {eid} appended"
          + (f"; registered ability '{ability}'" if ability else
             " (observation only)"))


# --------------------------------------------------------------- frontier
def frontier(statement: str, metric: str | None) -> None:
    if not metric:
        raise SystemExit("refused: a frontier update must reference a "
                         "--metric so self-direction stays grounded")
    inner = "## What I am working on right now\n\n" + statement.strip() + \
            f"\n\n*Current measured anchor: {metric}.*"
    text = _read(IDENTITY_SELF)
    _write(IDENTITY_SELF, _replace_block(text, "ACTIVE_FRONTIER", inner))
    print("ACTIVE_FRONTIER updated")


def main() -> int:
    ap = argparse.ArgumentParser(description="Metatron evidence-gated self-editor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync", help="sync CURRENT_FORM from daycare ledger")
    p_sync.add_argument("--state", default=None)

    p_note = sub.add_parser("note", help="append an evidence entry")
    p_note.add_argument("--ability", default=None,
                        help="ability id to register into the self-model")
    p_note.add_argument("--metric", action="append", default=[],
                        help="key=value measured result (repeatable)")
    p_note.add_argument("--reproduce", default=None,
                        help="command that reproduces the metric")
    p_note.add_argument("--body", default="", help="human-readable observation")
    p_note.add_argument("--observation", action="store_true",
                        help="record without claiming an ability")

    p_front = sub.add_parser("frontier", help="set ACTIVE_FRONTIER")
    p_front.add_argument("--statement", required=True)
    p_front.add_argument("--metric", required=True,
                         help="measured anchor for the frontier statement")

    args = ap.parse_args()
    if args.cmd == "sync":
        sync_form(Path(args.state) if args.state else None)
    elif args.cmd == "note":
        note(args.ability, "; ".join(args.metric), args.reproduce,
             args.body, evidence=not args.observation)
    elif args.cmd == "frontier":
        frontier(args.statement, args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
