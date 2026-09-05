#!/usr/bin/env python3
"""Deterministic benchmark gate for a persisted Metatron candidate."""
from __future__ import annotations
import argparse,importlib.util,json,os,pickle,sys,zipfile
from pathlib import Path

def find_source(root: Path) -> Path:
    c=list(root.rglob('metatron_v2.py'))
    if not c:
        for z in root.rglob('*grok-workspace.zip'):
            try:
                d=root/'_workspace_source'; d.mkdir(exist_ok=True)
                with zipfile.ZipFile(z) as f: f.extractall(d)
                c=list(d.rglob('metatron_v2.py'))
                if c: break
            except zipfile.BadZipFile: pass
    if not c: raise FileNotFoundError('metatron_v2.py not found')
    return c[0]

def load_source(root):
    path=find_source(root); name='metatron_v2_real'
    if name in sys.modules: return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--checkpoint',default=''); args=ap.parse_args()
    path=args.checkpoint or os.path.join(os.getenv('METATRON_CHECKPOINT_DIR',''),'model.pkl')
    if not path or not os.path.isfile(path): raise SystemExit('candidate checkpoint missing: '+path)
    load_source(Path(args.root).resolve())
    with open(path,'rb') as f: model=pickle.load(f)
    results=model.verify_capabilities(); passed=sum(bool(v) for v in results.values()); total=len(results)
    score=passed/max(1,total); metrics={f'cap_{k}':float(v) for k,v in results.items()}; metrics['capability_score']=score
    print(json.dumps({'score':score,'metrics':metrics}))
if __name__=='__main__': main()
