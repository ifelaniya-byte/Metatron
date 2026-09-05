#!/usr/bin/env python3
"""Low-energy, provenance-first corpus ingestion.

Sources are explicit URLs. Content is capped, hashed, and stored with source
metadata. This tool does not bypass access controls or silently scrape private
content. Review licenses before using collected text for training.
"""
from __future__ import annotations
import argparse, hashlib, json, time, urllib.request
from pathlib import Path

def fetch(url, cap):
    req=urllib.request.Request(url,headers={'User-Agent':'Metatron-Daycare/1.0'})
    with urllib.request.urlopen(req,timeout=20) as r: data=r.read(cap+1)
    return data[:cap]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--url',action='append',default=[]); ap.add_argument('--out',default='daycare_state/corpus.jsonl'); ap.add_argument('--cap',type=int,default=200_000); args=ap.parse_args()
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:
      for url in args.url:
        try:
          b=fetch(url,args.cap); text=b.decode('utf-8','replace'); sha=hashlib.sha256(b).hexdigest()
          row={'ts':time.time(),'source_url':url,'sha256':sha,'bytes':len(b),'license':'REVIEW_REQUIRED','text':text}
          f.write(json.dumps(row,ensure_ascii=False)+'\n'); print(json.dumps({'ok':True,'url':url,'sha256':sha,'bytes':len(b)}))
        except Exception as e: print(json.dumps({'ok':False,'url':url,'error':str(e)}))
if __name__=='__main__': main()
