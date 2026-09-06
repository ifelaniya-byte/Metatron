"""Tests for Metatron's evidence-gated self-editor.

The self-copy (`self/IDENTITY_SELF.md`, `self/JOURNAL.md`) is Metatron's to
update, but only under the charter rule: an ability enters the self-model
with a measured metric and a reproduction command; unsupported claims are
refused; the anchored charters are never writable through this path.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "self_editor", ROOT / "daycare" / "self_editor.py")
se = importlib.util.module_from_spec(spec)
sys.modules["self_editor"] = se
spec.loader.exec_module(se)


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Point the editor at an isolated self/ tree built from the committed seed."""
    d = tmp_path / "self"
    d.mkdir()
    (d / "IDENTITY_SELF.md").write_text(
        (ROOT / "self" / "IDENTITY_SELF.md").read_text(encoding="utf-8"))
    (d / "JOURNAL.md").write_text(
        (ROOT / "self" / "JOURNAL.md").read_text(encoding="utf-8"))
    monkeypatch.setattr(se, "SELF_DIR", d)
    monkeypatch.setattr(se, "IDENTITY_SELF", d / "IDENTITY_SELF.md")
    monkeypatch.setattr(se, "JOURNAL", d / "JOURNAL.md")
    return d


def _block(text, name):
    m = re.search(re.escape(f"<!-- BEGIN:{name} -->")
                  + r"(.*?)" + re.escape(f"<!-- END:{name} -->"), text, re.DOTALL)
    return m.group(1) if m else ""


def test_note_without_metric_is_refused(sandbox):
    with pytest.raises(SystemExit):
        se.note(ability="fluent-english", metric=None, reproduce=None,
                body="i feel fluent", evidence=True)


def test_note_ability_without_reproduce_is_refused(sandbox):
    with pytest.raises(SystemExit):
        se.note(ability="arithmetic", metric="acc 0.97", reproduce=None,
                body="math", evidence=True)


def test_proven_ability_enters_self_model(sandbox):
    se.note(ability="digit-addition",
            metric="held-out digit addition acc 0.96 (n=31)",
            reproduce="python daycare/math_literacy.py",
            body="I answer held-out addition facts correctly.",
            evidence=True)
    abilities = _block((sandbox / "IDENTITY_SELF.md").read_text(), "PROVEN_ABILITIES")
    assert "digit-addition" in abilities
    assert "0.96" in abilities
    journal = (sandbox / "JOURNAL.md").read_text()
    assert "ABILITY: digit-addition" in journal
    assert "REPRODUCE:" in journal


def test_observation_without_ability_does_not_register(tmp_path, monkeypatch):
    # Pristine tree (no abilities pre-registered) to prove an --observation
    # entry does not add one even when it carries a metric.
    d = tmp_path / "self"
    d.mkdir()
    minimal_id = (
        "<!-- BEGIN:PROVEN_ABILITIES -->\n"
        "## What I have proven I can do\n\n"
        "- *(none recorded yet)*\n"
        "<!-- END:PROVEN_ABILITIES -->\n")
    (d / "IDENTITY_SELF.md").write_text(minimal_id)
    (d / "JOURNAL.md").write_text("# JOURNAL\n\n## Entries\n")
    monkeypatch.setattr(se, "SELF_DIR", d)
    monkeypatch.setattr(se, "IDENTITY_SELF", d / "IDENTITY_SELF.md")
    monkeypatch.setattr(se, "JOURNAL", d / "JOURNAL.md")

    se.note(ability=None, metric="acc 0.10 (wall)", reproduce="x",
            body="I am stuck on operand conditioning.", evidence=False)
    abilities = _block((d / "IDENTITY_SELF.md").read_text(), "PROVEN_ABILITIES")
    assert "none recorded" in abilities
    assert "digit-addition" not in abilities


def test_entry_ids_are_unique(sandbox):
    for i in range(3):
        se.note(ability=f"ability-{i}", metric=f"m{i}=1", reproduce="r",
                body="b", evidence=True)
    ids = re.findall(r"^###\s*(\S+)",
                     (sandbox / "JOURNAL.md").read_text(), re.MULTILINE)
    assert len(ids) == len(set(ids)), f"duplicate journal ids: {ids}"


def test_frontier_requires_a_metric(sandbox):
    with pytest.raises(SystemExit):
        se.frontier(statement="I will learn calculus.", metric=None)
    se.frontier(statement="Drilling identity facts x+0=x first.",
                metric="identity accuracy 0/6 (not yet learned)")
    block = _block((sandbox / "IDENTITY_SELF.md").read_text(), "ACTIVE_FRONTIER")
    assert "identity facts" in block


def test_charter_files_are_protected():
    # The editor constants point at the self/ copies, never the charters.
    assert se.IDENTITY_SELF.name == "IDENTITY_SELF.md"
    assert "SELF_IDENTITY.md" in se.PROTECTED
    assert "LEARNING_MISSION.md" in se.PROTECTED
