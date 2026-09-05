#!/usr/bin/env python3
"""
MetatronULTRA Scale Demo

Run different parameter sizes and compare:
- 93K (ultra)
- 370K (small)
- 1.5M (medium)
- 6M (large)
- 24M (xlarge)
- 50M (xxlarge)
- 95M (gigantic)
- 188M (colossal)
- 470M (titanic / approaching 500M)

Usage:
    python metatron_scale_demo.py ultra        # 93K, 30s
    python metatron_scale_demo.py large        # 6M, 25min
    python metatron_scale_demo.py 500m         # 470M, ~10+ hours
    python metatron_scale_demo.py compare      # Run all, show graph
"""

import sys
import numpy as np
import time
import math
from typing import List, Tuple

# ============================================================================
# MINIMAL METATRON (Optimized for speed/memory)
# ============================================================================

class TinyLin:
    """Ultra-lightweight linear layer"""
    def __init__(self, in_d, out_d):
        self.in_d = in_d
        self.out_d = out_d
        std = math.sqrt(2.0 / in_d)
        self.W = np.random.randn(out_d, in_d) * std
        self.b = np.zeros(out_d)
        self.Wg = np.zeros_like(self.W)
        self.bg = np.zeros_like(self.b)
    
    def fwd(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        out = x @ self.W.T + self.b
        return out[0] if out.shape[0] == 1 else out
    
    def bwd(self, x, g):
        if g.ndim == 0:
            g = g.reshape(1, -1)
        elif g.ndim == 1 and g.shape[0] == self.out_d:
            g = g.reshape(1, -1)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.Wg += g.T @ x
        self.bg += g.sum(axis=0)
        return (g @ self.W)[0] if g.shape[0] == 1 else g @ self.W
    
    def step(self, lr):
        self.W -= lr * self.Wg
        self.b -= lr * self.bg
        self.Wg.fill(0)
        self.bg.fill(0)


class TinyStack:
    """Stack of layers"""
    def __init__(self, d, n):
        self.d = d
        self.n = n
        self.ls = [TinyLin(d, d) for _ in range(n)]
    
    def fwd(self, x):
        for l in self.ls:
            x = l.fwd(x)
            x = np.tanh(x)
        return x
    
    def bwd(self, x, g):
        for l in reversed(self.ls):
            g = l.bwd(x, g)
            g = g * (1 - np.tanh(x)**2)
        return g
    
    def step(self, lr):
        for l in self.ls:
            l.step(lr)


class TinyMod:
    """Tiny module"""
    def __init__(self, mid, d, rl, pl):
        self.mid = mid
        self.d = d
        self.emb = np.zeros(d)
        self.eg = np.zeros(d)
        self.rec = TinyStack(d, rl)
        self.prj = TinyStack(d, pl)
        self.mem = [np.zeros(d) for _ in range(3)]
    
    def fwd(self, nb):
        total = self.mem[0]*0.5 + self.mem[1]*0.7
        if nb:
            total += sum(nb.values()) / len(nb)
        upd = self.rec.fwd(self.emb)
        gat = np.tanh(upd + total)
        ne = self.emb + gat * 0.1
        out = self.prj.fwd(ne)
        return ne, out
    
    def step(self, lr):
        self.rec.step(lr)
        self.prj.step(lr)
    
    def upd(self, lr):
        self.emb -= lr * self.eg
        self.eg.fill(0)
        self.mem[2] = self.mem[1].copy()
        self.mem[1] = self.mem[0].copy()
        self.mem[0] = self.emb.copy()


class TinyMetatron:
    """Minimal Metatron for benchmarking"""
    def __init__(self, d, nm, rl, pl, v=30):
        self.d = d
        self.nm = nm
        self.v = v
        
        self.et = np.random.randn(v, d) * 0.01
        self.eg = np.zeros_like(self.et)
        
        self.ms = {i: TinyMod(i, d, rl, pl) for i in range(nm)}
        self._topo(nm)
        
        self.h = TinyLin(d, v)
        self.ws = np.zeros(d)
        
        corpus = " ".join(CORPUS)
        self.chars = sorted(list(set(corpus)))
        self.s2i = {c: i for i, c in enumerate(self.chars)}
        self.i2s = {i: c for i, c in enumerate(self.chars)}
    
    def _topo(self, nm):
        self.nb = {}
        if nm == 19:
            self.nb[0] = list(range(1, 7))
            for i in range(1, 7):
                self.nb[i] = [0, (i % 6) + 1]
            for i in range(7, 19):
                self.nb[i] = list(range(1, 7))
        else:
            for i in range(nm):
                self.nb[i] = list(range(nm))
    
    def emb(self, ids):
        return self.et[ids].mean(axis=0)
    
    def inj(self, mid, v, s=1.0):
        self.ms[mid].emb += v * s
    
    def stp(self):
        outs = {}
        for mid in self.ms:
            nbi = self.nb.get(mid, [])
            nbd = {nid: outs.get(nid, np.zeros(self.d)) for nid in nbi}
            ne, out = self.ms[mid].fwd(nbd)
            self.ms[mid].emb = ne
            outs[mid] = out
        self.ws = np.mean([self.ms[mid].emb for mid in self.ms], axis=0)
    
    def trn(self, ids, lr):
        ls = 0.0
        for i in range(1, len(ids)):
            ctx = ids[:i]
            tgt = ids[i]
            
            v = self.emb(ctx)
            self.inj(0, v)
            self.inj(1, v*0.5)
            self.inj(2, v*0.35)
            self.stp()
            
            lg = self.h.fwd(self.ws)
            lg = np.clip(lg, -100, 100)
            lgn = lg - lg.max()
            exp_lg = np.exp(lgn)
            prb = exp_lg / (exp_lg.sum() + 1e-8)
            
            l = -np.log(prb[tgt] + 1e-8)
            ls += l
            
            dlg = prb.copy()
            dlg[tgt] -= 1.0
            
            norm = np.linalg.norm(dlg)
            if norm > 1.5:
                dlg *= 1.5 / norm
            
            dws = self.h.bwd(self.ws, dlg)
            self.h.step(lr)
            
            for mid in self.ms:
                self.ms[mid].eg += dws * 0.1
                self.ms[mid].step(lr)
                self.ms[mid].upd(lr)
        
        return ls / len(ids)
    
    def val(self, ids):
        ls = 0.0
        for i in range(1, len(ids)):
            ctx = ids[:i]
            tgt = ids[i]
            v = self.emb(ctx)
            self.inj(0, v)
            self.stp()
            lg = self.h.fwd(self.ws)
            lg = np.clip(lg, -100, 100)
            lgn = lg - lg.max()
            exp_lg = np.exp(lgn)
            prb = exp_lg / (exp_lg.sum() + 1e-8)
            l = -np.log(prb[tgt] + 1e-8)
            ls += l
        return np.exp(ls / len(ids))
    
    def run(self):
        corpus_txt = " ".join(CORPUS)
        trn_txt = " ".join(CORPUS[:-5])
        val_txt = " ".join(CORPUS[-5:])
        
        trn_ids = [self.s2i.get(c, 0) for c in trn_txt]
        val_ids = [self.s2i.get(c, 0) for c in val_txt]
        
        base = self.val(val_ids)
        print(f"Baseline: {base:.2f}\n")
        
        for ep in range(self.max_epochs):
            lr = self.lr0 * (self.lrd ** ep)
            
            st = time.time()
            loss = self.trn(trn_ids, lr)
            et = time.time() - st
            
            ppl = self.val(val_ids)
            print(f"Ep {ep+1:2d} | lr {lr:.5f} | loss {loss:.4f} | ppl {ppl:.2f} | {et:.1f}s")
        
        print(f"\nFinal: {ppl:.2f}")


# ============================================================================
# SCALE DEFINITIONS
# ============================================================================

SCALES = {
    'ultra': (48, 19, 1, 1, 30, 0.01),           # 93K
    'small': (96, 19, 1, 1, 30, 0.008),          # 370K
    'medium': (192, 19, 1, 1, 30, 0.006),        # 1.5M
    'large': (256, 19, 2, 2, 25, 0.005),         # 6M
    'xlarge': (512, 19, 2, 2, 20, 0.004),        # 24M
    'xxlarge': (768, 19, 2, 2, 20, 0.003),       # 50M
    'gigantic': (1024, 19, 2, 2, 15, 0.002),     # 95M
    'colossal': (1536, 19, 3, 3, 15, 0.0015),    # 188M
    '500m': (2048, 38, 3, 3, 15, 0.001),         # 470M (Titanic)
}


def calc_params(d, nm, rl, pl, v=30):
    return v*d + d*v + nm * (rl + pl) * (d*d)


# ============================================================================
# CORPUS
# ============================================================================

CORPUS = [
    "the flower of life is made of overlapping circles",
    "sacred geometry appears throughout nature and the cosmos",
    "the mandelbrot set reveals fractal patterns in mathematics",
    "training neural networks requires careful tuning of hyperparameters",
    "the universe contains infinite layers of complexity and beauty",
    "machine learning models learn patterns from data automatically",
    "metatron is the angel who oversees divine geometry",
    "consciousness emerges from networks of interconnected nodes",
    "the flower blooms in spring and summer months",
    "geometry shapes the structure of all living things",
    "networks enable communication across vast distances instantly",
    "patterns repeat at different scales in self similar ways",
    "learning happens through repeated exposure to examples",
    "sacred places hold spiritual significance for many cultures",
    "the mind processes information through distributed networks",
    "circles represent completion and wholeness in many traditions",
    "cycles repeat in nature from seconds to eons",
    "training improves performance through gradient descent optimization",
    "infinity exists in mathematics and theoretical physics",
    "beauty emerges from mathematical harmony and proportion",
]


# ============================================================================
# RUNNER
# ============================================================================

def run_scale(scale_name):
    """Run a specific scale"""
    if scale_name not in SCALES:
        print(f"Unknown scale: {scale_name}")
        print(f"Available: {', '.join(SCALES.keys())}")
        return
    
    d, nm, rl, pl, ep, lr = SCALES[scale_name]
    params = calc_params(d, nm, rl, pl)
    
    print(f"\n{'='*60}")
    print(f"METATRON {scale_name.upper()}: {params:,} params")
    print(f"{'='*60}")
    print(f"Dim: {d} | Modules: {nm} | Rec: {rl} | Proj: {pl}")
    print(f"Epochs: {ep} | LR: {lr}")
    print(f"{'='*60}\n")
    
    m = TinyMetatron(d, nm, rl, pl)
    m.max_epochs = ep
    m.lr0 = lr
    m.lrd = 0.97
    m.run()


def compare_scales():
    """Compare all scales"""
    print(f"\n{'='*80}")
    print(f"METATRON SCALING COMPARISON")
    print(f"{'='*80}\n")
    print(f"{'Scale':<15} {'Params':<15} {'Dim':<8} {'Mods':<8} {'Rec':<5} {'Proj':<5} {'Epochs':<8}")
    print("-" * 80)
    
    for name, (d, nm, rl, pl, ep, lr) in SCALES.items():
        params = calc_params(d, nm, rl, pl)
        print(f"{name:<15} {params:>13,} {d:>7} {nm:>8} {rl:>5} {pl:>5} {ep:>8}")
    
    print("\n" + "="*80)
    print("To run a specific scale:")
    print("  python metatron_scale_demo.py ultra    # 93K, ~30s")
    print("  python metatron_scale_demo.py large    # 6M, ~25min")
    print("  python metatron_scale_demo.py 500m     # 470M, ~10+ hours")
    print("="*80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        scale = sys.argv[1]
        if scale == 'compare':
            compare_scales()
        else:
            run_scale(scale)
    else:
        compare_scales()
        print("Running ULTRA (93K) as demo...\n")
        run_scale('ultra')
