#!/usr/bin/env python3
"""Frontier-teacher arena for Metatron.

This does not claim that a small local model beats frontier models. Instead it
uses frontier models as adversarial teachers/judges and converts only verified,
measurable gains into Metatron's durable experience ledger.
"""
from __future__ import annotations
import json, os, time, urllib.request
from pathlib import Path

MODEL = os.getenv("METATRON_FRONTIER_MODEL", "claude-fable-5-1")
API_URL = os.getenv("METATRON_FRONTIER_URL", "https://api.anthropic.com/v1/messages")
KEY = os.getenv("ANTHROPIC_API_KEY", "")

SYSTEM = """You are a frontier benchmark teacher for Metatron. Be adversarial and precise.
Find the smallest reproducible task where the candidate fails, explain the root cause,
propose a training example or software change, and define a deterministic acceptance test.
Never declare victory without an objective test."""

def ask(prompt: str) -> dict:
    if not KEY:
        return {"status": "skipped", "reason": "ANTHROPIC_API_KEY not configured"}
    body=json.dumps({"model":MODEL,"max_tokens":4096,"system":SYSTEM,"messages":[{"role":"user","content":prompt}]}).encode()
    req=urllib.request.Request(API_URL, data=body, headers={"content-type":"application/json","x-api-key":KEY,"anthropic-version":"2023-06-01"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        data=json.load(r)
    return {"status":"ok","model":MODEL,"response":data,"timestamp":time.time()}

def main() -> None:
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--prompt",required=True); ap.add_argument("--out",default="daycare_state/frontier"); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    result=ask(a.prompt)
    (out/"latest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"model":MODEL}))

if __name__ == "__main__": main()
