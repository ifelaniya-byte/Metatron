#!/usr/bin/env python3
"""Optional OpenAI-compatible teacher ensemble with provider rotation."""
from __future__ import annotations
import argparse,json,os,urllib.request
from pathlib import Path

def call(base,key,model,prompt):
    url=base.rstrip('/')+'/chat/completions'; body=json.dumps({'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0.2,'max_tokens':700}).encode()
    req=urllib.request.Request(url,data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=90) as r: d=json.load(r)
    return d['choices'][0]['message']['content']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prompt',required=True); ap.add_argument('--out',default='daycare_state/teacher.jsonl'); args=ap.parse_args()
    providers=json.loads(os.getenv('METATRON_TEACHERS','[]')); p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
    if not providers: raise SystemExit('METATRON_TEACHERS must be JSON list of {base_url,api_key_env,model}')
    for x in providers:
      try:
        text=call(x['base_url'],os.getenv(x['api_key_env'],''),x['model'],args.prompt)
        with p.open('a',encoding='utf-8') as f: f.write(json.dumps({'provider':x['base_url'],'model':x['model'],'prompt':args.prompt,'response':text,'provenance':'teacher-model-output'})+'\n')
        print(json.dumps({'ok':True,'model':x['model']})); return
      except Exception as e: print(json.dumps({'ok':False,'model':x.get('model'),'error':str(e)}))
    raise SystemExit(1)
if __name__=='__main__': main()
