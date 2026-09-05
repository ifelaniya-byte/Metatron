#!/usr/bin/env python3
"""Real Metatron trainer adapter with full-state, shutdown-safe checkpoints."""
from __future__ import annotations
import argparse, importlib.util, json, os, pickle, sys, zipfile
from pathlib import Path

def find_source(root: Path) -> Path:
    candidates=list(root.rglob('metatron_v2.py'))
    if not candidates:
        for outer in root.rglob('*grok-workspace.zip'):
            try:
                d=root/'_workspace_source'; d.mkdir(exist_ok=True)
                with zipfile.ZipFile(outer) as zf: zf.extractall(d)
                candidates=list(d.rglob('metatron_v2.py'))
                if candidates: break
            except zipfile.BadZipFile: pass
    if not candidates:
        for z in root.rglob('Metatron_Full_System.zip'):
            try:
                d=root/'_metatron_source'; d.mkdir(exist_ok=True)
                with zipfile.ZipFile(z) as zf: zf.extractall(d)
                candidates=list(d.rglob('metatron_v2.py'))
                if candidates: break
            except zipfile.BadZipFile: pass
    if not candidates: raise FileNotFoundError('metatron_v2.py not found in checkout/workspace')
    return candidates[0]

def load_module(path: Path):
    spec=importlib.util.spec_from_file_location('metatron_v2_real',path)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--scale',default=os.getenv('METATRON_SCALE','ultra')); ap.add_argument('--epochs',type=int,default=None); ap.add_argument('--text',default=os.getenv('METATRON_TEXT','')); ap.add_argument('--checkpoint',default=''); ap.add_argument('--parent',default=''); args=ap.parse_args()
    root=Path(args.root).resolve(); src=find_source(root); mod=load_module(src)
    ck=Path(args.checkpoint or (os.getenv('METATRON_CHECKPOINT_DIR','')+'/model.pkl')); ck.parent.mkdir(parents=True,exist_ok=True)
    parent=Path(args.parent) if args.parent else root/'daycare_state/champion/model.pkl'
    model=pickle.load(open(parent,'rb')) if parent.is_file() else mod.MetatronV2(mod.SCALES[args.scale])
    # The released architecture is numerically unstable at its original LR.
    # Daycare deliberately starts at a conservative LR; promotion gates can
    # later justify a controlled increase.
    model.cfg.learning_rate=min(float(model.cfg.learning_rate),float(os.getenv('METATRON_SAFE_LR','0.0001')))
    model.cfg.grad_clip=min(float(model.cfg.grad_clip),float(os.getenv('METATRON_GRAD_CLIP','0.5')))
    text=Path(args.text).read_text(encoding='utf-8') if args.text and Path(args.text).is_file() else mod.DEFAULT_TEXT
    epochs=max(1,args.epochs if args.epochs is not None else int(os.getenv('METATRON_BUDGET','25'))//25)
    mod.train(model,text,epochs=epochs,out_dir=str(ck.parent/'numpy_diagnostics'))
    tmp=ck.with_suffix(ck.suffix+'.tmp')
    with open(tmp,'wb') as f: pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,ck)
    print(json.dumps({'status':'ok','checkpoint':str(ck),'scale':args.scale,'epochs':epochs,'source':str(src)}))
if __name__=='__main__': main()
