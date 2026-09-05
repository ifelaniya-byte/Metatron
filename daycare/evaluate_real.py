#!/usr/bin/env python3
"""Deterministic proxy evaluation for a persisted Metatron candidate."""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); args=ap.parse_args()
    with open(args.checkpoint,'rb') as f: model=pickle.load(f)
    results=model.verify_capabilities()
    passed=sum(bool(v) for v in results.values()); total=len(results)
    score=passed/max(1,total)
    metrics={f'cap_{k}':float(v) for k,v in results.items()}; metrics['capability_score']=score
    print(json.dumps({'score':score,'metrics':metrics}))
if __name__=='__main__': main()
