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

def load_source(root: Path):
    name='metatron_v2_real'
    if name in sys.modules: return sys.modules[name]
    p=find_source(root); spec=importlib.util.spec_from_file_location(name,p); mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--checkpoint',default=''); ap.add_argument('--text',default=''); args=ap.parse_args()
    path=args.checkpoint or os.path.join(os.getenv('METATRON_CHECKPOINT_DIR',''),'model.pkl')
    if not path or not os.path.isfile(path): raise SystemExit('candidate checkpoint missing: '+path)
    mod=load_source(Path(args.root).resolve())
    with open(path,'rb') as f: model=pickle.load(f)
    caps=mod.verify_capabilities(model); passed=sum(bool(v) for v in caps.values()); total=len(caps)
    ids=model.encode(Path(args.text).read_text(encoding='utf-8') if args.text and Path(args.text).is_file() else mod.DEFAULT_TEXT,add_bos=True,add_eos=True)
    loss=float(model.loss(ids))
    capability_score=passed/max(1,total)
    score=-loss if passed==total else float('-inf')
    metrics={f'cap_{k}':float(v) for k,v in caps.items()}; metrics.update({'capability_score':capability_score,'eval_loss':loss,'eval_perplexity':min(1e12,__import__('math').exp(min(loss,27)))})
    print(json.dumps({'score':score,'metrics':metrics}))
if __name__=='__main__': main()
