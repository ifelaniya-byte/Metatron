#!/usr/bin/env python3
"""Metatron Daycare: low-energy, resumable training/evolution orchestrator.

The daycare is intentionally a controller, not a new neural architecture. It
keeps durable state, chooses the cheapest useful experiment, runs a configured
trainer, evaluates candidates, and promotes only benchmark-winning checkpoints.

For true "learn while my laptop is off", run this worker on a remote machine
(or CI/self-hosted GPU). The laptop is only a controller/client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = 1

@dataclass
class Candidate:
    candidate_id: str
    parent: str
    kind: str
    budget: int
    reason: str
    checkpoint: str = ""
    score: float = 0.0
    status: str = "queued"
    metrics: dict[str, float] = field(default_factory=dict)

@dataclass
class State:
    schema: int = SCHEMA
    generation: int = 0
    xp: int = 0
    level: int = 1
    best_score: float = float("-inf")
    champion: str = ""
    active: str = ""
    energy_spent: float = 0.0
    candidates: list[Candidate] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

class Daycare:
    def __init__(self, root: Path):
        self.root = root
        self.state_path = root / "state.json"
        self.ledger_path = root / "experience.jsonl"
        self.queue_path = root / "queue.jsonl"
        self.ckpt_dir = root / "checkpoints"
        self.champion_dir = root / "champion"
        for p in (root, self.ckpt_dir, self.champion_dir):
            p.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> State:
        if not self.state_path.exists():
            return State()
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        raw["candidates"] = [Candidate(**c) for c in raw.get("candidates", [])]
        return State(**raw)

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def log(self, event: str, **data: Any) -> None:
        row = {"ts": time.time(), "event": event, **data}
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        self.state.history.append(row)
        self.save()

    def enqueue(self, candidate: Candidate) -> None:
        with self.queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(candidate), sort_keys=True) + "\n")
        self.state.candidates.append(candidate)
        self.save()

    def next_candidate(self) -> Candidate | None:
        for c in self.state.candidates:
            if c.status == "queued":
                return c
        return None

    def hatch(self, parent: str, budget: int = 100, reason: str = "curriculum") -> Candidate:
        material = f"{parent}|{self.state.generation}|{time.time_ns()}".encode()
        cid = hashlib.sha256(material).hexdigest()[:12]
        # Prefer cheap changes first: data/curriculum > LoRA > architecture.
        kind = "curriculum" if self.state.generation % 4 else "lora"
        c = Candidate(cid, parent, kind, max(1, budget), reason)
        self.enqueue(c)
        self.log("egg_hatched", candidate=cid, parent=parent, kind=kind, budget=c.budget)
        return c

    def run(self, command: str, candidate: Candidate) -> int:
        env = os.environ.copy()
        env.update({
            "METATRON_DAYCARE": "1",
            "METATRON_CANDIDATE": candidate.candidate_id,
            "METATRON_PARENT": candidate.parent,
            "METATRON_KIND": candidate.kind,
            "METATRON_BUDGET": str(candidate.budget),
            "METATRON_CHECKPOINT_DIR": str(self.ckpt_dir / candidate.candidate_id),
        })
        Path(env["METATRON_CHECKPOINT_DIR"]).mkdir(parents=True, exist_ok=True)
        argv = shlex.split(command, posix=(os.name != "nt"))
        self.log("train_start", candidate=candidate.candidate_id, argv=argv)
        started = time.time()
        proc = subprocess.run(argv, env=env)
        elapsed = time.time() - started
        self.state.energy_spent += max(0.0, elapsed)
        self.log("train_end", candidate=candidate.candidate_id, returncode=proc.returncode, seconds=elapsed)
        return proc.returncode

    def evaluate(self, command: str, candidate: Candidate) -> tuple[float, dict[str, float]]:
        env = os.environ.copy()
        env.update({
            "METATRON_DAYCARE": "1",
            "METATRON_CANDIDATE": candidate.candidate_id,
            "METATRON_PARENT": candidate.parent,
            "METATRON_CHECKPOINT_DIR": str(self.ckpt_dir / candidate.candidate_id),
        })
        argv = shlex.split(command, posix=(os.name != "nt"))
        self.log("eval_start", candidate=candidate.candidate_id)
        out = subprocess.run(argv, env=env, capture_output=True, text=True)
        if out.returncode != 0:
            self.log("eval_failed", candidate=candidate.candidate_id, stderr=out.stderr[-2000:])
            return float("-inf"), {}
        # Evaluator must print JSON: {"score": number, "metrics": {...}}.
        try:
            payload = json.loads(out.stdout.strip().splitlines()[-1])
            score = float(payload["score"])
            metrics = {k: float(v) for k, v in payload.get("metrics", {}).items()}
        except (ValueError, KeyError, json.JSONDecodeError):
            self.log("eval_bad_output", candidate=candidate.candidate_id, stdout=out.stdout[-2000:])
            return float("-inf"), {}
        self.log("eval_end", candidate=candidate.candidate_id, score=score, metrics=metrics)
        return score, metrics

    def promote(self, candidate: Candidate, score: float, metrics: dict[str, float], min_gain: float) -> bool:
        candidate.score, candidate.metrics = score, metrics
        old = self.state.best_score
        accepted = score > old + min_gain
        if accepted:
            source = self.ckpt_dir / candidate.candidate_id
            if source.exists():
                tmp = self.champion_dir.with_name("champion.tmp")
                if tmp.exists(): shutil.rmtree(tmp)
                shutil.copytree(source, tmp)
                if self.champion_dir.exists(): shutil.rmtree(self.champion_dir)
                tmp.replace(self.champion_dir)
            self.state.best_score = score
            self.state.champion = candidate.candidate_id
            candidate.status = "champion"
            self.state.generation += 1
            self.state.xp += max(1, int(score * 100))
            self.state.level = 1 + self.state.xp // 1000
            self.log("promoted", candidate=candidate.candidate_id, score=score, previous=old)
        else:
            candidate.status = "rejected"
            self.log("rejected", candidate=candidate.candidate_id, score=score, best=old)
        self.save()
        return accepted

    def package_ollama(self, modelfile: Path, tag: str) -> int:
        if shutil.which("ollama") is None:
            self.log("ollama_skip", reason="ollama executable not found", tag=tag)
            return 127
        self.log("ollama_create", tag=tag)
        return subprocess.run(["ollama", "create", tag, "-f", str(modelfile)]).returncode

    def choose_budget(self, max_budget: int) -> int:
        # Energy-aware hill climbing: spend a small budget until the last
        # accepted improvement justifies more compute.
        if self.state.best_score == float("-inf"):
            return min(max_budget, 25)
        recent = [h for h in self.state.history[-8:] if h["event"] == "promoted"]
        if not recent:
            return min(max_budget, 25)
        return min(max_budget, 50 if len(recent) >= 2 else 25)


def main() -> int:
    ap = argparse.ArgumentParser(description="Metatron Pokemon Daycare")
    ap.add_argument("--root", default=os.getenv("METATRON_DAYCARE_ROOT", "./daycare_state"))
    ap.add_argument("--trainer", default=os.getenv("METATRON_TRAINER", ""), help="Trainer command")
    ap.add_argument("--evaluator", default=os.getenv("METATRON_EVALUATOR", ""), help="Evaluator command")
    ap.add_argument("--parent", default=os.getenv("METATRON_PARENT", "champion"))
    ap.add_argument("--budget", type=int, default=int(os.getenv("METATRON_MAX_BUDGET", "50")))
    ap.add_argument("--min-gain", type=float, default=float(os.getenv("METATRON_MIN_GAIN", "0.002")))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dc = Daycare(Path(args.root))
    if not dc.state.candidates:
        parent = dc.state.champion or args.parent
        dc.hatch(parent, dc.choose_budget(args.budget), "initial low-energy curriculum")

    while True:
        candidate = dc.next_candidate()
        if candidate is None:
            parent = dc.state.champion or args.parent
            dc.hatch(parent, dc.choose_budget(args.budget), "adaptive continuation")
            candidate = dc.next_candidate()
        assert candidate is not None
        candidate.status = "training"
        dc.save()

        if args.dry_run:
            dc.log("dry_run", candidate=candidate.candidate_id)
            candidate.status = "queued"
        elif not args.trainer or not args.evaluator:
            dc.log("blocked", candidate=candidate.candidate_id, reason="trainer/evaluator not configured")
            candidate.status = "blocked"
            dc.save()
            return 2
        else:
            rc = dc.run(args.trainer, candidate)
            if rc != 0:
                candidate.status = "failed"
                dc.save()
            else:
                score, metrics = dc.evaluate(args.evaluator, candidate)
                dc.promote(candidate, score, metrics, args.min_gain)

        dc.save()
        if args.once:
            return 0
        # Event-driven idle interval: no busy spin. A remote worker can stay
        # alive cheaply or be replaced by a scheduled invocation.
        time.sleep(max(10, int(os.getenv("METATRON_IDLE_SECONDS", "60"))))

if __name__ == "__main__":
    raise SystemExit(main())
