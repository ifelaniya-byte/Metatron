#!/usr/bin/env python3
"""
Metatron DENSE — Overstuffed Small Skeleton
===========================================
Keep the structural footprint as small as possible (low dim, few modules)
while jamming the maximum number of parameters into every layer.

Strategy:
- Tiny dim (32-64) and modest module count
- Extremely wide SwiGLU (8x-12x)
- Extra projection matrices per block
- More residual blocks stacked inside each module
- Parameter sharing only where it hurts quality least
- Still pure geometric Flower-of-Life routing

Goal: highest params-per-structure ratio possible while remaining runnable on 8GB.
"""

import numpy as np
import math
import time
import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)

def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def clip_grad(g, max_norm=1.0):
    norm = np.sqrt(np.sum(g * g) + 1e-12)
    return g * (max_norm / norm) if norm > max_norm else g

# =============================================================================
# CONFIG — deliberately small skeleton, dense insides
# =============================================================================

@dataclass
class DenseConfig:
    name: str = "dense-ultra"
    dim: int = 48                 # keep small
    n_modules: int = 13           # keep modest
    n_heads: int = 4
    n_layers: int = 3             # stack depth inside each module
    ffn_mult: float = 10.0        # OVERSTUFF: normal is ~2.7-4, we go 8-12
    extra_proj: int = 2           # extra linear maps per block
    vocab_size: int = 256
    context_length: int = 64
    batch_size: int = 6
    learning_rate: float = 0.0025
    weight_decay: float = 0.02
    max_epochs: int = 30
    grad_clip: float = 0.8

    @property
    def total_params(self) -> int:
        emb = self.vocab_size * self.dim
        head = self.dim * self.vocab_size
        hidden = int(self.dim * self.ffn_mult)
        # per block: qkv + out + extra projs + swiglu + router + norms
        per_block = (
            3 * self.dim * self.dim +          # qkv
            self.dim * self.dim +              # out
            self.extra_proj * self.dim * self.dim +
            2 * self.dim * hidden +            # swiglu gates
            hidden * self.dim +                # swiglu down
            self.n_modules +                   # router
            2 * self.dim                       # norm scales
        )
        return emb + head + self.n_modules * self.n_layers * per_block

# Pre-defined overstuffed packs (smallest skeleton first)
DENSE_SCALES = {
    "nano": DenseConfig(
        name="nano", dim=32, n_modules=7, n_layers=3, ffn_mult=12.0, extra_proj=2,
        vocab_size=128, context_length=48, batch_size=8, learning_rate=0.004
    ),
    "micro": DenseConfig(
        name="micro", dim=40, n_modules=9, n_layers=3, ffn_mult=11.0, extra_proj=2,
        vocab_size=192, context_length=56, batch_size=6, learning_rate=0.0035
    ),
    "dense-ultra": DenseConfig(
        name="dense-ultra", dim=48, n_modules=13, n_layers=3, ffn_mult=10.0, extra_proj=2,
        vocab_size=256, context_length=64, batch_size=6, learning_rate=0.0025
    ),
    "dense-tiny": DenseConfig(
        name="dense-tiny", dim=56, n_modules=13, n_layers=4, ffn_mult=9.0, extra_proj=3,
        vocab_size=256, context_length=80, batch_size=4, learning_rate=0.002
    ),
    "dense-small": DenseConfig(
        name="dense-small", dim=64, n_modules=19, n_layers=3, ffn_mult=8.0, extra_proj=2,
        vocab_size=320, context_length=96, batch_size=3, learning_rate=0.0018
    ),
}

# =============================================================================
# LAYERS
# =============================================================================

class Linear:
    def __init__(self, in_f, out_f, bias=True):
        std = math.sqrt(2.0 / (in_f + out_f))
        self.W = (np.random.randn(out_f, in_f) * std).astype(np.float32)
        self.b = np.zeros(out_f, dtype=np.float32) if bias else None
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b) if bias else None
        self.vb = np.zeros_like(self.b) if bias else None
        self.t = 0
        self._x = None
        self._dW = None
        self._db = None

    def forward(self, x):
        self._x = x
        y = x @ self.W.T
        if self.b is not None:
            y = y + self.b
        return y

    def backward(self, dy):
        x = self._x
        if x.ndim == 1:
            self._dW = np.outer(dy, x)
            dx = self.W.T @ dy
            self._db = dy if self.b is not None else None
        else:
            self._dW = dy.T @ x
            dx = dy @ self.W
            self._db = dy.sum(0) if self.b is not None else None
        return dx

    def step(self, lr, clip=1.0, wd=0.01, beta1=0.9, beta2=0.95, eps=1e-8):
        if self._dW is None:
            return
        self.t += 1
        g = clip_grad(self._dW, clip)
        if wd:
            g = g + wd * self.W
        self.mW = beta1 * self.mW + (1 - beta1) * g
        self.vW = beta2 * self.vW + (1 - beta2) * (g * g)
        mhat = self.mW / (1 - beta1 ** self.t)
        vhat = self.vW / (1 - beta2 ** self.t)
        self.W -= lr * mhat / (np.sqrt(vhat) + eps)
        if self.b is not None and self._db is not None:
            gb = clip_grad(self._db, clip)
            self.mb = beta1 * self.mb + (1 - beta1) * gb
            self.vb = beta2 * self.vb + (1 - beta2) * (gb * gb)
            self.b -= lr * (self.mb / (1 - beta1 ** self.t)) / (np.sqrt(self.vb / (1 - beta2 ** self.t)) + eps)
        self._dW = None
        self._db = None


class DenseBlock:
    """Overstuffed residual block: multi-head geometric attn + wide SwiGLU + extra projs"""
    def __init__(self, cfg: DenseConfig):
        d = cfg.dim
        self.n_heads = cfg.n_heads
        self.head_dim = d // cfg.n_heads
        self.n_modules = cfg.n_modules

        self.Wq = Linear(d, d, bias=False)
        self.Wk = Linear(d, d, bias=False)
        self.Wv = Linear(d, d, bias=False)
        self.Wo = Linear(d, d)
        self.router = Linear(d, cfg.n_modules)

        # Extra projection matrices (the "stuffing")
        self.extra = [Linear(d, d) for _ in range(cfg.extra_proj)]

        # Wide SwiGLU
        hidden = int(d * cfg.ffn_mult)
        self.w1 = Linear(d, hidden, bias=False)
        self.w2 = Linear(d, hidden, bias=False)
        self.w3 = Linear(hidden, d, bias=False)

        self.n1 = np.ones(d, dtype=np.float32)
        self.n2 = np.ones(d, dtype=np.float32)

    def forward(self, x, states: List[np.ndarray]):
        # Pre-norm + geometric multi-head
        h = layer_norm(x) * self.n1
        q = self.Wq.forward(h).reshape(self.n_heads, self.head_dim)
        K = np.stack([self.Wk.forward(s) for s in states]).reshape(self.n_modules, self.n_heads, self.head_dim)
        V = np.stack([self.Wv.forward(s) for s in states]).reshape(self.n_modules, self.n_heads, self.head_dim)

        scores = np.einsum("hd,mhd->hm", q, K) / math.sqrt(self.head_dim)
        scores = scores + self.router.forward(h)[None, :]
        attn = softmax(scores, axis=-1)
        out = np.einsum("hm,mhd->hd", attn, V).reshape(-1)
        out = self.Wo.forward(out)

        # Extra dense projections (overstuff)
        for proj in self.extra:
            out = out + 0.3 * proj.forward(out)

        x = x + out

        # Pre-norm + wide SwiGLU
        h = layer_norm(x) * self.n2
        x = x + self.w3.forward(silu(self.w1.forward(h)) * self.w2.forward(h))
        return x

    def step(self, lr, clip, wd):
        for layer in [self.Wq, self.Wk, self.Wv, self.Wo, self.router,
                      self.w1, self.w2, self.w3] + self.extra:
            layer.step(lr, clip, wd)


# =============================================================================
# MODEL
# =============================================================================

class MetatronDense:
    def __init__(self, cfg: DenseConfig):
        self.cfg = cfg
        self.dim = cfg.dim
        self.n_modules = cfg.n_modules

        std = 0.02
        self.embed = (np.random.randn(cfg.vocab_size, cfg.dim) * std).astype(np.float32)
        self.pos = (np.random.randn(cfg.context_length, cfg.dim) * std).astype(np.float32)
        self.head = Linear(cfg.dim, cfg.vocab_size)
        self.final_scale = np.ones(cfg.dim, dtype=np.float32)

        # Each module is a stack of dense blocks
        self.modules = [
            [DenseBlock(cfg) for _ in range(cfg.n_layers)]
            for _ in range(cfg.n_modules)
        ]

        # Flower topology scaffold
        self.topology = self._flower(cfg.n_modules)
        self.tok = self._tokenizer(cfg.vocab_size)
        self.t = 0

    def _flower(self, n):
        topo = [[] for _ in range(n)]
        for i in range(n):
            for k in (1, 2, 3, 5, 8):
                topo[i].append((i + k) % n)
                topo[i].append((i - k) % n)
        return topo

    def _tokenizer(self, vs):
        chars = [chr(i) for i in range(32, 127)]
        special = ["<pad>", "<unk>", "<bos>", "<eos>"]
        itos = (special + chars + [f"<e{i}>" for i in range(vs)])[:vs]
        stoi = {c: i for i, c in enumerate(itos)}
        return {"itos": itos, "stoi": stoi, "pad": 0, "unk": 1, "bos": 2, "eos": 3}

    def encode(self, text, bos=True, eos=False):
        ids = [self.tok["bos"]] if bos else []
        ids += [self.tok["stoi"].get(c, self.tok["unk"]) for c in text]
        if eos:
            ids.append(self.tok["eos"])
        return ids

    def decode(self, ids):
        return "".join(
            self.tok["itos"][i] for i in ids
            if 0 <= i < len(self.tok["itos"]) and not self.tok["itos"][i].startswith("<")
        )

    def forward(self, ids: List[int]) -> np.ndarray:
        states = [np.zeros(self.dim, np.float32) for _ in range(self.n_modules)]
        logits = []
        for t, tid in enumerate(ids):
            x = self.embed[tid] + self.pos[t % self.cfg.context_length]
            for m in range(self.n_modules):
                for block in self.modules[m]:
                    x = block.forward(x, states)
                states[m] = 0.85 * states[m] + 0.15 * x
            h = layer_norm(x) * self.final_scale
            logits.append(self.head.forward(h))
        return np.stack(logits)

    def loss(self, ids: List[int]) -> float:
        if len(ids) < 2:
            return 0.0
        logits = self.forward(ids[:-1])
        loss = 0.0
        for t, target in enumerate(ids[1:]):
            logit = logits[t] - np.max(logits[t])
            probs = np.exp(logit)
            probs /= probs.sum() + 1e-8
            loss += -np.log(probs[target] + 1e-8)
            d = probs.copy()
            d[target] -= 1.0
            self.head.backward(d)
        return loss / (len(ids) - 1)

    def step(self, lr=None):
        lr = lr or self.cfg.learning_rate
        self.t += 1
        self.head.step(lr, self.cfg.grad_clip, self.cfg.weight_decay)
        for mod in self.modules:
            for block in mod:
                block.step(lr, self.cfg.grad_clip, self.cfg.weight_decay)
        self.embed *= (1.0 - lr * self.cfg.weight_decay * 0.1)

    def generate(self, prompt, max_new=60, temperature=0.8, top_k=30):
        ids = self.encode(prompt, bos=True)
        for _ in range(max_new):
            ctx = ids[-self.cfg.context_length:]
            logit = self.forward(ctx)[-1] / max(temperature, 1e-5)
            logit -= np.max(logit)
            p = np.exp(logit)
            p /= p.sum() + 1e-8
            if top_k < len(p):
                idx = np.argpartition(p, -top_k)[-top_k:]
                mask = np.zeros_like(p)
                mask[idx] = p[idx]
                p = mask / (mask.sum() + 1e-8)
            nid = int(np.random.choice(len(p), p=p))
            if nid == self.tok["eos"]:
                break
            ids.append(nid)
        return self.decode(ids)

    def save(self, path):
        try:
            data = {"embed": self.embed, "pos": self.pos,
                    "head_W": self.head.W, "head_b": self.head.b,
                    "final_scale": self.final_scale}
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **data)
            os.replace(tmp, path)
            with open(path + ".meta.json", "w") as f:
                json.dump({"config": asdict(self.cfg), "params": self.cfg.total_params}, f, indent=2)
            print(f"Saved → {path} ({self.cfg.total_params:,} params)")
        except Exception as e:
            print(f"[warn] save: {e}")

    def load(self, path):
        npz = np.load(path)
        self.embed = npz["embed"]
        self.pos = npz["pos"]
        self.head.W = npz["head_W"]
        self.head.b = npz["head_b"]
        self.final_scale = npz["final_scale"]
        print(f"Loaded ← {path}")


# =============================================================================
# TRAIN + REPORT
# =============================================================================

TEXT = """
the flower of life is made of overlapping circles in sacred geometry
sacred geometry appears throughout nature and the cosmos in many forms
metatron is the angel who oversees the flow of divine energy and geometry
neural networks learn patterns from data through gradient descent optimization
language models predict the next token given previous context and training
the universe contains infinite layers of complexity beauty and structure
consciousness emerges from networks of interconnected processing nodes
machine learning models improve by minimizing prediction error over time
patterns repeat at different scales in self similar fractal ways
training improves performance when learning rate and architecture are balanced
a small language model can still be useful if trained carefully on good data
metatron grows from a simple flower of life into a functioning reasoning engine
transformers use self attention while metatron uses geometric message passing
residual connections allow deep networks to train without vanishing gradients
layer normalization stabilizes the distribution of activations during training
multi head attention lets the model look at information from different subspaces
overstuffing parameters into a tiny skeleton tests information density limits
dense geometric modules aim to beat transformers on params per structure
""".strip()

def train(model, text, epochs, out_dir="metatron_dense_ckpts"):
    cfg = model.cfg
    os.makedirs(out_dir, exist_ok=True)
    ids = model.encode(text, bos=True, eos=True)
    ctx = cfg.context_length
    windows = [ids[i:i+ctx+1] for i in range(0, max(1, len(ids)-ctx-1), max(1, ctx//2))]
    if not windows:
        windows = [ids]
    print(f"\n{'='*60}")
    print(f"  OVERSTUFFED Metatron — {cfg.name}")
    print(f"  skeleton: dim={cfg.dim}  modules={cfg.n_modules}  layers/mod={cfg.n_layers}")
    print(f"  stuffing: ffn_mult={cfg.ffn_mult}x  extra_proj={cfg.extra_proj}")
    print(f"  TOTAL PARAMS: {cfg.total_params:,}")
    print(f"{'='*60}")
    print(f"Data windows: {len(windows)}")

    lr = cfg.learning_rate
    best = 1e9
    for ep in range(epochs):
        t0 = time.time()
        total, n = 0.0, 0
        np.random.shuffle(windows)
        for seq in windows:
            loss = model.loss(seq)
            model.step(lr)
            total += loss * (len(seq)-1)
            n += len(seq)-1
        avg = total / max(1, n)
        ppl = math.exp(min(avg, 20))
        lr *= 0.98
        print(f"Epoch {ep+1:3d}/{epochs} | loss {avg:.4f} | ppl {ppl:.1f} | "
              f"lr {lr:.5f} | {time.time()-t0:.1f}s")
        if avg < best:
            best = avg
            model.save(os.path.join(out_dir, f"{cfg.name}_best.npz"))
        if (ep+1) % 4 == 0 or ep == epochs-1:
            print(f"  gen: {model.generate('the flower', max_new=40, temperature=0.7)[:70]}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="nano", choices=list(DENSE_SCALES.keys()))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--generate", default=None)
    parser.add_argument("--load", default=None)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.report:
        print("\nOverstuff density report (params / structural unit):\n")
        for name, cfg in DENSE_SCALES.items():
            struct = cfg.dim * cfg.n_modules * cfg.n_layers
            density = cfg.total_params / max(1, struct)
            print(f"  {name:14s}  params={cfg.total_params:9,}  "
                  f"skeleton={cfg.dim}d×{cfg.n_modules}m×{cfg.n_layers}L  "
                  f"density={density:,.0f} params/unit")
        return

    cfg = DENSE_SCALES[args.scale]
    model = MetatronDense(cfg)
    if args.load and os.path.isfile(args.load):
        model.load(args.load)
    if args.generate is not None:
        print(model.generate(args.generate, max_new=80))
        return

    train(model, TEXT, args.epochs)
    print("\nFinal sample:", model.generate("the flower of life", max_new=50))

if __name__ == "__main__":
    main()
