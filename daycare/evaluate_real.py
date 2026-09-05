#!/usr/bin/env python3
"""Deterministic benchmark gate for a persisted Metatron candidate."""
from __future__ import annotations
import argparse,json,os,pickle

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=''); args=ap.parse_args()
    path=args.checkpoint or os.path.join(os.getenv('METATRON_CHECKPOINT_DIR',''),'model.pkl')
    if not path or not os.path.isfile(path): raise SystemExit('candidate checkpoint missing: '+path)
    with open(path,'rb') as f: model=pickle.load(f)
    results=model.verify_capabilities(); passed=sum(bool(v) for v in results.values()); total=len(results)
    score=passed/max(1,total); metrics={f'cap_{k}':float(v) for k,v in results.items()}; metrics['capability_score']=score
    print(json.dumps({'score':score,'metrics':metrics}))
if __name__=='__main__': main()
