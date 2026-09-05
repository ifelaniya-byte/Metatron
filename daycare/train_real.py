#!/usr/bin/env python3
"""Train a real Metatron_v2 candidate and persist the entire model object.
The upstream release's save() omits internal module weights; this adapter uses
pickle so the full numpy model + optimizer state survives shutdowns."""
from __future__ import annotations
import argparse, importlib.util, json, os, pickle, sys, zipfile
from pathlib import Path

def find_source(root: Path) -> Path:
    candidates = list(root.rglob('metatron_v2.py'))
    if not candidates:
        zips = list(root.rglob('Metatron_Full_System.zip'))
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                names=[n for n in zf.namelist() if n.endswith('metatron_v2.py')]
                if names:
                    d=root/'_metatron_source'; d.mkdir(exist_ok=True)
                    zf.extractall(d); candidates=list(d.rglob('metatron_v2.py')); break
    if not candidates: raise FileNotFoundError('metatron_v2.py / Metatron_Full_System.zip not found')
    return candidates[0]

def load_module(path: Path):
    spec=importlib.util.spec_from_file_location('metatron_v2_real', path)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--scale',default='ultra'); ap.add_argument('--epochs',type=int,default=1); ap.add_argument('--text',default=''); ap.add_argument('--checkpoint',required=True); ap.add_argument('--parent',default=''); args=ap.parse_args()
    root=Path(args.root).resolve(); src=find_source(root); mod=load_module(src)
    ck=Path(args.checkpoint); ck.parent.mkdir(parents=True,exist_ok=True)
    if args.parent and Path(args.parent).is_file():
        with open(args.parent,'rb') as f: model=pickle.load(f)
    else: model=mod.MetatronV2(mod.SCALES[args.scale])
    text=Path(args.text).read_text(encoding='utf-8') if args.text and Path(args.text).is_file() else mod.DEFAULT_TEXT
    mod.train(model,text,epochs=max(1,args.epochs),out_dir=str(ck.parent/'numpy_diagnostics'))
    tmp=ck.with_suffix(ck.suffix+'.tmp');
    with open(tmp,'wb') as f: pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,ck)
    print(json.dumps({'status':'ok','checkpoint':str(ck),'scale':args.scale,'epochs':args.epochs,'source':str(src)}))
if __name__=='__main__': main()
