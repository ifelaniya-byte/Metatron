#!/usr/bin/env python3
"""
Metatron BODY — Full Organ System
=================================
Gives the geometric core a complete agent body:

  Brain   = dense geometric reasoning core
  Memory  = short-term buffer + long-term key-value store
  Ears    = input encoder (text → residual stream)
  Mouth   = generation head (residual stream → text)
  Hands   = tool / action interface
  Heart   = persistent identity + value vectors (bias every decision)
  Legs    = planner that turns goals into step sequences

Still pure NumPy. Still Flower-of-Life at the center.
Designed to stay small while having every organ.
"""

import numpy as np
import math
import time
import os
import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)

def layer_norm(x, eps=1e-5):
    m = np.mean(x, axis=-1, keepdims=True)
    v = np.var(x, axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)

def clip_grad(g, max_norm=1.0):
    n = np.sqrt(np.sum(g*g) + 1e-12)
    return g * (max_norm / n) if n > max_norm else g

# -----------------------------------------------------------------------------
# Config — keep skeleton small, organs explicit
# -----------------------------------------------------------------------------
@dataclass
class BodyConfig:
    name: str = "body-nano"
    dim: int = 48
    n_modules: int = 9
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: float = 6.0
    vocab_size: int = 256
    context_length: int = 64
    memory_slots: int = 64          # long-term memory size
    short_memory: int = 16          # working memory length
    learning_rate: float = 0.003
    max_epochs: int = 20
    grad_clip: float = 1.0

    @property
    def total_params(self) -> int:
        d = self.dim
        emb = self.vocab_size * d
        head = d * self.vocab_size
        # core blocks
        hidden = int(d * self.ffn_mult)
        per_block = (4*d*d + 2*d*hidden + hidden*d + self.n_modules + 2*d)
        core = self.n_modules * self.n_layers * per_block
        # organs
        memory = self.memory_slots * d * 2          # keys + values
        heart = d * 4                               # identity + 3 value vectors
        hands = d * 32                              # tool embedding table (32 tools)
        legs = d * d * 2                            # planner
        return emb + head + core + memory + heart + hands + legs

# -----------------------------------------------------------------------------
# Core layers (Brain)
# -----------------------------------------------------------------------------
class Linear:
    def __init__(self, in_f, out_f, bias=True):
        std = math.sqrt(2.0 / (in_f + out_f))
        self.W = (np.random.randn(out_f, in_f) * std).astype(np.float32)
        self.b = np.zeros(out_f, np.float32) if bias else None
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b) if bias else None
        self.vb = np.zeros_like(self.b) if bias else None
        self.t = 0
        self._x = None
        self._dW = self._db = None

    def forward(self, x):
        self._x = x
        y = x @ self.W.T
        return y + self.b if self.b is not None else y

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

    def step(self, lr, clip=1.0, wd=0.01):
        if self._dW is None:
            return
        self.t += 1
        g = clip_grad(self._dW, clip) + wd * self.W
        self.mW = 0.9 * self.mW + 0.1 * g
        self.vW = 0.95 * self.vW + 0.05 * (g*g)
        self.W -= lr * (self.mW / (1-0.9**self.t)) / (np.sqrt(self.vW/(1-0.95**self.t)) + 1e-8)
        if self.b is not None and self._db is not None:
            gb = clip_grad(self._db, clip)
            self.mb = 0.9*self.mb + 0.1*gb
            self.vb = 0.95*self.vb + 0.05*(gb*gb)
            self.b -= lr * (self.mb/(1-0.9**self.t)) / (np.sqrt(self.vb/(1-0.95**self.t))+1e-8)
        self._dW = self._db = None


class BrainBlock:
    """Geometric multi-head block — the Brain tissue"""
    def __init__(self, cfg: BodyConfig):
        d = cfg.dim
        self.n_heads = cfg.n_heads
        self.head_dim = d // cfg.n_heads
        self.n_modules = cfg.n_modules
        self.Wq = Linear(d, d, bias=False)
        self.Wk = Linear(d, d, bias=False)
        self.Wv = Linear(d, d, bias=False)
        self.Wo = Linear(d, d)
        self.router = Linear(d, cfg.n_modules)
        hidden = int(d * cfg.ffn_mult)
        self.w1 = Linear(d, hidden, bias=False)
        self.w2 = Linear(d, hidden, bias=False)
        self.w3 = Linear(hidden, d, bias=False)
        self.n1 = np.ones(d, np.float32)
        self.n2 = np.ones(d, np.float32)

    def forward(self, x, states):
        h = layer_norm(x) * self.n1
        q = self.Wq.forward(h).reshape(self.n_heads, self.head_dim)
        K = np.stack([self.Wk.forward(s) for s in states]).reshape(self.n_modules, self.n_heads, self.head_dim)
        V = np.stack([self.Wv.forward(s) for s in states]).reshape(self.n_modules, self.n_heads, self.head_dim)
        scores = np.einsum("hd,mhd->hm", q, K) / math.sqrt(self.head_dim)
        scores = scores + self.router.forward(h)[None, :]
        attn = softmax(scores, -1)
        out = np.einsum("hm,mhd->hd", attn, V).reshape(-1)
        x = x + self.Wo.forward(out)
        h = layer_norm(x) * self.n2
        x = x + self.w3.forward(silu(self.w1.forward(h)) * self.w2.forward(h))
        return x

    def step(self, lr, clip):
        for l in [self.Wq, self.Wk, self.Wv, self.Wo, self.router, self.w1, self.w2, self.w3]:
            l.step(lr, clip)


# -----------------------------------------------------------------------------
# ORGANS
# -----------------------------------------------------------------------------
class Memory:
    """Long-term key-value memory + short-term working buffer"""
    def __init__(self, dim, n_slots, short_len):
        self.dim = dim
        self.keys = np.zeros((n_slots, dim), np.float32)
        self.values = np.zeros((n_slots, dim), np.float32)
        self.age = np.zeros(n_slots, np.float32)
        self.ptr = 0
        self.short = deque(maxlen=short_len)   # recent residual states
        self.write_proj = Linear(dim, dim)
        self.read_proj = Linear(dim, dim)

    def write(self, state, importance=1.0):
        """Write a state into long-term memory (circular + importance)"""
        k = self.write_proj.forward(state)
        # find least important / oldest slot
        score = self.age - 0.3 * np.linalg.norm(self.values, axis=1)
        idx = int(np.argmin(score)) if self.ptr >= len(self.keys) else self.ptr
        self.keys[idx] = k
        self.values[idx] = state * importance
        self.age[idx] = 0
        self.age += 1
        self.ptr = min(self.ptr + 1, len(self.keys))
        self.short.append(state.copy())

    def read(self, query, top_k=3):
        """Retrieve top-k relevant memories"""
        if self.ptr == 0:
            return np.zeros(self.dim, np.float32)
        q = self.read_proj.forward(query)
        sims = self.keys[:self.ptr] @ q
        k = min(top_k, self.ptr)
        idx = np.argpartition(sims, -k)[-k:]
        weights = softmax(sims[idx])
        return (weights[:, None] * self.values[idx]).sum(0)

    def short_context(self):
        if not self.short:
            return np.zeros(self.dim, np.float32)
        return np.mean(list(self.short), axis=0)


class Heart:
    """Identity + values that continuously bias the residual stream"""
    def __init__(self, dim):
        self.identity = (np.random.randn(dim) * 0.02).astype(np.float32)  # who I am
        self.values = (np.random.randn(3, dim) * 0.02).astype(np.float32) # core drives
        self.mood = np.zeros(dim, np.float32)
        self.names = ["curiosity", "helpfulness", "coherence"]

    def bias(self, state):
        """Return a heart bias vector to add to the residual stream"""
        # identity always present
        b = 0.15 * self.identity
        # value vectors modulated by current state similarity
        for i, v in enumerate(self.values):
            strength = float(np.dot(state, v) / (np.linalg.norm(state) * np.linalg.norm(v) + 1e-8))
            b = b + 0.08 * strength * v
        b = b + 0.05 * self.mood
        return b

    def update_mood(self, reward):
        self.mood = 0.9 * self.mood + 0.1 * reward * self.identity


class Hands:
    """Tool / action interface"""
    TOOLS = [
        "think", "recall", "plan", "speak", "search", "calculate",
        "write_memory", "read_memory", "set_goal", "finish",
        "wait", "observe", "compare", "summarize", "ask", "refuse"
    ]

    def __init__(self, dim, n_tools=32):
        self.dim = dim
        self.tool_emb = (np.random.randn(n_tools, dim) * 0.02).astype(np.float32)
        self.selector = Linear(dim, n_tools)
        self.n_tools = min(n_tools, len(self.TOOLS))

    def choose(self, state) -> Tuple[str, float]:
        logits = self.selector.forward(state)
        probs = softmax(logits[:self.n_tools])
        idx = int(np.argmax(probs))
        return self.TOOLS[idx], float(probs[idx])

    def embed(self, tool_name: str) -> np.ndarray:
        try:
            idx = self.TOOLS.index(tool_name)
        except ValueError:
            idx = 0
        return self.tool_emb[idx]


class Legs:
    """Planner — turns a goal into a sequence of steps"""
    def __init__(self, dim):
        self.plan_proj = Linear(dim, dim)
        self.step_head = Linear(dim, dim)
        self.current_plan: List[str] = []
        self.step_idx = 0

    def make_plan(self, goal_state, max_steps=5) -> List[str]:
        """Simple geometric planner: walk the residual in the direction of the goal"""
        h = self.plan_proj.forward(goal_state)
        steps = []
        for i in range(max_steps):
            h = np.tanh(self.step_head.forward(h))
            # map direction to a coarse action word
            score = h.sum()
            if score > 0.5:
                steps.append("act")
            elif score > 0:
                steps.append("think")
            elif score > -0.5:
                steps.append("recall")
            else:
                steps.append("observe")
        self.current_plan = steps
        self.step_idx = 0
        return steps

    def next_step(self) -> Optional[str]:
        if self.step_idx >= len(self.current_plan):
            return None
        s = self.current_plan[self.step_idx]
        self.step_idx += 1
        return s


class Ears:
    """Input encoder"""
    def __init__(self, dim, vocab_size, ctx):
        self.embed = (np.random.randn(vocab_size, dim) * 0.02).astype(np.float32)
        self.pos = (np.random.randn(ctx, dim) * 0.02).astype(np.float32)
        self.ctx = ctx

    def encode_tokens(self, ids: List[int]) -> np.ndarray:
        """Return the final residual after hearing the sequence (simple mean + last)"""
        if not ids:
            return np.zeros(self.embed.shape[1], np.float32)
        states = []
        for t, tid in enumerate(ids[-self.ctx:]):
            states.append(self.embed[tid] + self.pos[t % self.ctx])
        return 0.7 * states[-1] + 0.3 * np.mean(states, axis=0)


class Mouth:
    """Speech / generation head"""
    def __init__(self, dim, vocab_size):
        self.head = Linear(dim, vocab_size)
        self.dim = dim

    def logits(self, state):
        return self.head.forward(layer_norm(state))

    def speak(self, state, tok, temperature=0.8, top_k=30, max_new=40):
        ids = []
        h = state.copy()
        for _ in range(max_new):
            logit = self.logits(h) / max(temperature, 1e-5)
            logit -= np.max(logit)
            p = np.exp(logit)
            p /= p.sum() + 1e-8
            if top_k < len(p):
                idx = np.argpartition(p, -top_k)[-top_k:]
                mask = np.zeros_like(p)
                mask[idx] = p[idx]
                p = mask / (mask.sum() + 1e-8)
            nid = int(np.random.choice(len(p), p=p))
            if nid == tok["eos"]:
                break
            ids.append(nid)
            # crude state update
            h = 0.85 * h + 0.15 * np.random.randn(self.dim).astype(np.float32) * 0.01
        return ids


# -----------------------------------------------------------------------------
# FULL BODY
# -----------------------------------------------------------------------------
class MetatronBody:
    def __init__(self, cfg: BodyConfig):
        self.cfg = cfg
        self.dim = cfg.dim

        # Organs
        self.ears = Ears(cfg.dim, cfg.vocab_size, cfg.context_length)
        self.mouth = Mouth(cfg.dim, cfg.vocab_size)
        self.memory = Memory(cfg.dim, cfg.memory_slots, cfg.short_memory)
        self.heart = Heart(cfg.dim)
        self.hands = Hands(cfg.dim)
        self.legs = Legs(cfg.dim)

        # Brain (geometric core)
        self.brain = [
            [BrainBlock(cfg) for _ in range(cfg.n_layers)]
            for _ in range(cfg.n_modules)
        ]
        self.states = [np.zeros(cfg.dim, np.float32) for _ in range(cfg.n_modules)]

        # Tokenizer
        self.tok = self._build_tok(cfg.vocab_size)
        self.step_count = 0

    def _build_tok(self, vs):
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

    # ----- Brain forward -----
    def think(self, x):
        """Run residual stream through the geometric brain, biased by heart + memory"""
        # Heart bias
        x = x + self.heart.bias(x)
        # Memory injection
        x = x + 0.25 * self.memory.read(x)
        x = x + 0.15 * self.memory.short_context()

        for m in range(self.cfg.n_modules):
            for block in self.brain[m]:
                x = block.forward(x, self.states)
            self.states[m] = 0.88 * self.states[m] + 0.12 * x
        return x

    # ----- Full perception → action cycle -----
    def cycle(self, text_input: str, max_inner=3) -> Dict[str, Any]:
        """
        One full body cycle:
        Ears → Brain (+ Heart + Memory) → Legs (plan) → Hands (tool) → Mouth
        """
        log = {"input": text_input, "steps": []}

        # EARS
        ids = self.encode(text_input, bos=True)
        x = self.ears.encode_tokens(ids)
        log["steps"].append("ears: heard input")

        # BRAIN + HEART + MEMORY
        for _ in range(max_inner):
            x = self.think(x)
        self.memory.write(x, importance=1.0)
        log["steps"].append("brain+heart+memory: thought")

        # LEGS — make a plan
        plan = self.legs.make_plan(x, max_steps=4)
        log["plan"] = plan
        log["steps"].append(f"legs: plan={plan}")

        # HANDS — choose tools along the plan
        actions = []
        for step in plan:
            tool, conf = self.hands.choose(x)
            actions.append({"step": step, "tool": tool, "conf": round(conf, 3)})
            # tool embedding feeds back into state
            x = x + 0.2 * self.hands.embed(tool)
            x = self.think(x)
        log["actions"] = actions
        log["steps"].append("hands: acted")

        # MOUTH — speak
        out_ids = self.mouth.speak(x, self.tok, temperature=0.75, max_new=50)
        utterance = self.decode(out_ids)
        log["utterance"] = utterance
        log["steps"].append("mouth: spoke")

        self.step_count += 1
        return log

    def generate(self, prompt, max_new=60, temperature=0.8):
        ids = self.encode(prompt, bos=True)
        x = self.ears.encode_tokens(ids)
        x = self.think(x)
        out_ids = self.mouth.speak(x, self.tok, temperature=temperature, max_new=max_new)
        return self.decode(ids + out_ids)

    def train_step(self, text):
        """Simple next-token style training signal through the body"""
        ids = self.encode(text, bos=True, eos=True)
        if len(ids) < 3:
            return 0.0
        x = self.ears.encode_tokens(ids[:-1])
        x = self.think(x)
        logits = self.mouth.logits(x)
        target = ids[-1]
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum() + 1e-8
        loss = -np.log(probs[target] + 1e-8)
        # coarse gradient
        d = probs.copy()
        d[target] -= 1.0
        self.mouth.head.backward(d)
        self.mouth.head.step(self.cfg.learning_rate, self.cfg.grad_clip)
        for mod in self.brain:
            for b in mod:
                b.step(self.cfg.learning_rate, self.cfg.grad_clip)
        return float(loss)

    def save(self, path):
        try:
            data = {
                "ears_embed": self.ears.embed,
                "ears_pos": self.ears.pos,
                "mouth_W": self.mouth.head.W,
                "mouth_b": self.mouth.head.b,
                "heart_identity": self.heart.identity,
                "heart_values": self.heart.values,
                "memory_keys": self.memory.keys,
                "memory_values": self.memory.values,
            }
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **data)
            os.replace(tmp, path)
            meta = {"config": asdict(self.cfg), "params": self.cfg.total_params,
                    "step_count": self.step_count}
            with open(path + ".meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Saved body → {path} ({self.cfg.total_params:,} params)")
        except Exception as e:
            print(f"[warn] save: {e}")

    def status(self):
        return {
            "organs": ["brain", "memory", "ears", "mouth", "hands", "heart", "legs"],
            "params": self.cfg.total_params,
            "memory_filled": int(self.memory.ptr),
            "memory_slots": self.cfg.memory_slots,
            "short_memory": len(self.memory.short),
            "heart_drives": self.heart.names,
            "tools": self.hands.TOOLS[:self.hands.n_tools],
            "steps_lived": self.step_count,
        }


# -----------------------------------------------------------------------------
# Demo / train
# -----------------------------------------------------------------------------
DEMO_TEXTS = [
    "the flower of life contains all sacred geometry",
    "metatron remembers what it has learned",
    "a body needs brain heart hands legs mouth ears and memory",
    "I listen with ears and speak with mouth",
    "I plan with legs and act with hands",
    "my heart keeps my values stable while I think",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--cycle", type=str, default=None,
                        help="Run one full body cycle with this input")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--generate", type=str, default=None)
    args = parser.parse_args()

    cfg = BodyConfig()
    body = MetatronBody(cfg)

    print("\n" + "="*60)
    print("  METATRON BODY — all organs present")
    print("="*60)
    st = body.status()
    for k, v in st.items():
        print(f"  {k}: {v}")
    print("="*60)

    if args.status:
        return

    if args.cycle:
        log = body.cycle(args.cycle)
        print(json.dumps(log, indent=2))
        return

    if args.generate:
        print(body.generate(args.generate))
        return

    # short training + live cycles
    print("\nTraining body...")
    for ep in range(args.epochs):
        losses = [body.train_step(t) for t in DEMO_TEXTS]
        avg = sum(losses) / len(losses)
        print(f"Epoch {ep+1}/{args.epochs} | loss {avg:.4f}")

    body.save("metatron_body_ckpts/body_nano.npz")

    print("\n--- Full body cycle demo ---")
    log = body.cycle("what are you and what can you do?")
    print("Input :", log["input"])
    print("Plan  :", log["plan"])
    print("Actions:")
    for a in log["actions"]:
        print(f"  {a}")
    print("Mouth :", log["utterance"])
    print("\nOrgans used:", " → ".join(log["steps"]))

if __name__ == "__main__":
    main()
