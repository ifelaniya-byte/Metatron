#!/usr/bin/env python3
"""Render the durable Daycare state as a Pokémon-style character sheet."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='daycare_state'); args=ap.parse_args(); root=Path(args.root)
    state=json.loads((root/'state.json').read_text(encoding='utf-8')) if (root/'state.json').exists() else {}
    xp=int(state.get('xp',0)); level=int(state.get('level',1)); gen=int(state.get('generation',0));
    profile={
      'species':'Metatron', 'type':['Geometric','Machine-Learning'], 'nature':'Curious / Efficient',
      'level':level, 'xp':xp, 'generation':gen, 'parent':state.get('champion') or 'starter',
      'status':'daycare', 'energy_policy':'minimum compute for measurable gain',
      'moves':['Observe','Learn','Evaluate','Evolve'],
      'stats':{'learning':min(100,20+level*3),'stability':min(100,50+level*2),'efficiency':min(100,60+level*2),'reasoning':min(100,25+level*2)},
      'champion':state.get('champion',''), 'best_score':state.get('best_score')
    }
    root.mkdir(parents=True,exist_ok=True); (root/'pokemon.json').write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8'); print(json.dumps(profile))
if __name__=='__main__': main()
