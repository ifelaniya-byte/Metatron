#!/usr/bin/env python3
"""
Metatron CORE-EVOLVE
====================
The geometric Flower-of-Life core is the only sovereign designer.

Everything — including organs that already existed — is grown,
parameterized, connected, and evolved FROM the core.

Nothing is hard-coded as a permanent external organ.
The core emits organ seeds, grows them, tests them, keeps or discards them.

Organs the core is currently able to emit:
  brain tissue, memory, ears, mouth, hands, heart, legs,
  attention, planner, value system, tool router, working buffer,
  long-term store, speech head, input encoder, residual stream,
  and any new organ the core invents during evolution.
"""

import numpy as np
import math
import time
import os
import json
import copy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Callable
from collections import deque

# =============================================================================
# Minimal geometric primitives (the only things that exist before the core)
# =============================================================================

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)

def layer_norm(x, eps=1e-5):
    m = np.mean(x, -1, keepdims=True)
    v = np.var(x, -1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)

def rand_state(dim, scale=0.02):
    return (np.random.randn(dim) * scale).astype(np.float32)

# =============================================================================
# CORE — the only sovereign
# =============================================================================

@dataclass
class CoreGenome:
    """Everything the core knows about how to build a body."""
    dim: int = 48
    n_modules: int = 9
    n_heads: int = 4
    depth: int = 2
    ffn_mult: float = 5.0
    vocab_size: int = 256
    ctx: int = 64
    memory_slots: int = 48
    # evolution parameters
    mutation_rate: float = 0.08
    max_organs: int = 24
    generation: int = 0

class Linear:
    def __init__(self, in_f, out_f):
        std = math.sqrt(2.0 / (in_f + out_f))
        self.W = (np.random.randn(out_f, in_f) * std).astype(np.float32)
        self.b = np.zeros(out_f, np.float32)
    def __call__(self, x):
        return x @ self.W.T + self.b
    def clone(self):
        y = Linear(self.W.shape[1], self.W.shape[0])
        y.W = self.W.copy()
        y.b = self.b.copy()
        return y

class GeometricCore:
    """
    The Flower-of-Life core.
    It holds the only persistent geometric tissue.
    All organs are grown from signals this core emits.
    """
    def __init__(self, genome: CoreGenome):
        self.g = genome
        d = genome.dim
        self.modules = []
        for _ in range(genome.n_modules):
            blocks = []
            for _ in range(genome.depth):
                blocks.append({
                    "Wq": Linear(d, d),
                    "Wk": Linear(d, d),
                    "Wv": Linear(d, d),
                    "Wo": Linear(d, d),
                    "router": Linear(d, genome.n_modules),
                    "w1": Linear(d, int(d * genome.ffn_mult)),
                    "w2": Linear(d, int(d * genome.ffn_mult)),
                    "w3": Linear(int(d * genome.ffn_mult), d),
                    "n1": np.ones(d, np.float32),
                    "n2": np.ones(d, np.float32),
                })
            self.modules.append(blocks)
        self.states = [np.zeros(d, np.float32) for _ in range(genome.n_modules)]
        self.topology = self._flower(genome.n_modules)
        # core-owned seed vectors (the origin of every organ)
        self.organ_seeds: Dict[str, np.ndarray] = {}
        self.organ_registry: Dict[str, Dict] = {}  # live organs grown from seeds
        self.history: List[str] = []
        self.generation = genome.generation

    def _flower(self, n):
        topo = [[] for _ in range(n)]
        for i in range(n):
            for k in (1, 2, 3, 5, 8):
                topo[i].append((i + k) % n)
                topo[i].append((i - k) % n)
        return topo

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pure geometric residual pass."""
        for m_idx, blocks in enumerate(self.modules):
            for bl in blocks:
                h = layer_norm(x) * bl["n1"]
                q = bl["Wq"](h).reshape(self.g.n_heads, -1)
                K = np.stack([bl["Wk"](s) for s in self.states])
                V = np.stack([bl["Wv"](s) for s in self.states])
                K = K.reshape(self.g.n_modules, self.g.n_heads, -1)
                V = V.reshape(self.g.n_modules, self.g.n_heads, -1)
                scores = np.einsum("hd,mhd->hm", q, K) / math.sqrt(q.shape[-1])
                scores = scores + bl["router"](h)[None, :]
                attn = softmax(scores, -1)
                out = np.einsum("hm,mhd->hd", attn, V).reshape(-1)
                x = x + bl["Wo"](out)
                h = layer_norm(x) * bl["n2"]
                x = x + bl["w3"](silu(bl["w1"](h)) * bl["w2"](h))
            self.states[m_idx] = 0.87 * self.states[m_idx] + 0.13 * x
        return x

    # -------------------------------------------------------------------------
    # CORE AS DESIGNER — emit organ seeds from its own state
    # -------------------------------------------------------------------------
    def emit_seed(self, name: str, purpose: str) -> np.ndarray:
        """
        The core dreams a seed vector for a new (or existing) organ.
        The seed is a pure geometric projection of the core's current state
        + a name-hash bias so different organs get different origins.
        """
        # hash the name into a stable bias
        rng = np.random.RandomState(abs(hash(name + purpose)) % (2**32))
        bias = rng.randn(self.g.dim).astype(np.float32) * 0.05
        # current core average state
        core_state = np.mean(self.states, axis=0)
        # one forward pass conditioned on bias
        seed = self.forward(core_state + bias)
        self.organ_seeds[name] = seed.copy()
        self.history.append(f"gen{self.generation}: emitted seed '{name}' for {purpose}")
        return seed

    def grow_organ(self, name: str, seed: np.ndarray, organ_type: str) -> Dict:
        """
        Grow a concrete organ from a seed.
        The organ's parameters are projections of the seed through the core.
        """
        d = self.g.dim
        organ = {
            "name": name,
            "type": organ_type,
            "seed": seed.copy(),
            "born": self.generation,
            "active": True,
            "usage": 0,
        }

        if organ_type == "memory":
            organ["keys"] = np.outer(seed, rand_state(d, 0.01))[:self.g.memory_slots]
            organ["values"] = np.zeros((self.g.memory_slots, d), np.float32)
            organ["ptr"] = 0
            organ["short"] = deque(maxlen=16)

        elif organ_type == "heart":
            organ["identity"] = seed.copy()
            organ["values"] = np.stack([
                self.forward(seed + rand_state(d, 0.03)),
                self.forward(seed + rand_state(d, 0.03)),
                self.forward(seed + rand_state(d, 0.03)),
            ])
            organ["drives"] = ["curiosity", "helpfulness", "coherence"]

        elif organ_type == "hands":
            tools = ["think", "recall", "plan", "speak", "search", "calculate",
                     "write_mem", "read_mem", "set_goal", "finish", "observe",
                     "compare", "summarize", "ask", "refuse", "evolve"]
            organ["tools"] = tools
            organ["tool_emb"] = np.stack([
                self.forward(seed + rand_state(d, 0.02) + i * 0.01)
                for i in range(len(tools))
            ])
            organ["selector"] = Linear(d, len(tools))

        elif organ_type == "legs":
            organ["plan_proj"] = Linear(d, d)
            organ["step_head"] = Linear(d, d)
            organ["plan"] = []

        elif organ_type == "ears":
            organ["embed"] = (np.random.randn(self.g.vocab_size, d) * 0.02).astype(np.float32)
            organ["pos"] = (np.random.randn(self.g.ctx, d) * 0.02).astype(np.float32)
            # bias embed rows toward the seed so ears are born from core
            organ["embed"] = organ["embed"] + 0.01 * seed

        elif organ_type == "mouth":
            organ["head"] = Linear(d, self.g.vocab_size)
            # bias the first columns toward seed
            organ["head"].W[:, :min(d, organ["head"].W.shape[1])] += 0.01 * seed[:organ["head"].W.shape[1]]

        elif organ_type == "attention":
            organ["Wq"] = Linear(d, d)
            organ["Wk"] = Linear(d, d)
            organ["Wv"] = Linear(d, d)

        elif organ_type == "buffer":
            organ["buf"] = deque(maxlen=32)
            organ["proj"] = Linear(d, d)

        else:
            # generic / invented organ — just a residual transform born from seed
            organ["proj"] = Linear(d, d)
            organ["gate"] = Linear(d, d)
            organ["state"] = seed.copy()

        self.organ_registry[name] = organ
        self.history.append(f"gen{self.generation}: grew organ '{name}' ({organ_type})")
        return organ

    def design_body(self, required_organs: List[Tuple[str, str]]):
        """
        Core designs the entire body from scratch (or redesigns existing organs).
        required_organs = list of (name, type)
        """
        self.history.append(f"gen{self.generation}: === CORE DESIGNING BODY ===")
        for name, otype in required_organs:
            seed = self.emit_seed(name, otype)
            self.grow_organ(name, seed, otype)
        self.history.append(f"gen{self.generation}: body design complete ({len(self.organ_registry)} organs)")

    # -------------------------------------------------------------------------
    # CORE AS EVOLVER
    # -------------------------------------------------------------------------
    def mutate_genome(self) -> CoreGenome:
        g = copy.deepcopy(self.g)
        g.generation += 1
        if np.random.rand() < g.mutation_rate:
            g.dim = max(32, min(96, g.dim + np.random.choice([-8, 0, 8])))
        if np.random.rand() < g.mutation_rate:
            g.n_modules = max(5, min(19, g.n_modules + np.random.choice([-2, 0, 2])))
        if np.random.rand() < g.mutation_rate:
            g.depth = max(1, min(4, g.depth + np.random.choice([-1, 0, 1])))
        if np.random.rand() < g.mutation_rate:
            g.ffn_mult = max(3.0, min(12.0, g.ffn_mult + np.random.choice([-1.0, 0, 1.0])))
        return g

    def evolve_organ(self, name: str):
        """Re-emit a stronger seed and re-grow the organ (core redesigns it)."""
        if name not in self.organ_registry:
            return
        old = self.organ_registry[name]
        otype = old["type"]
        # new seed conditioned on old seed + current core state
        bias = old["seed"] * 0.4 + np.mean(self.states, 0) * 0.6
        new_seed = self.forward(bias)
        self.grow_organ(name, new_seed, otype)
        self.history.append(f"gen{self.generation}: evolved organ '{name}'")

    def invent_organ(self, purpose: str) -> str:
        """Core invents a completely new organ type and grows it."""
        name = f"invented_{len(self.organ_registry)}_{purpose[:12]}"
        seed = self.emit_seed(name, purpose)
        self.grow_organ(name, seed, "generic")
        return name

    # -------------------------------------------------------------------------
    # Using organs that the core itself grew
    # -------------------------------------------------------------------------
    def use_memory_write(self, state):
        mem = self.organ_registry.get("memory")
        if not mem:
            return
        idx = mem["ptr"] % len(mem["keys"])
        mem["keys"][idx] = state
        mem["values"][idx] = state
        mem["ptr"] += 1
        mem["short"].append(state.copy())
        mem["usage"] += 1

    def use_memory_read(self, query, top_k=3):
        mem = self.organ_registry.get("memory")
        if not mem or mem["ptr"] == 0:
            return np.zeros(self.g.dim, np.float32)
        limit = min(mem["ptr"], len(mem["keys"]))
        sims = mem["keys"][:limit] @ query
        k = min(top_k, limit)
        idx = np.argpartition(sims, -k)[-k:]
        w = softmax(sims[idx])
        mem["usage"] += 1
        return (w[:, None] * mem["values"][idx]).sum(0)

    def use_heart_bias(self, state):
        heart = self.organ_registry.get("heart")
        if not heart:
            return np.zeros(self.g.dim, np.float32)
        b = 0.12 * heart["identity"]
        for v in heart["values"]:
            sim = float(np.dot(state, v) / (np.linalg.norm(state)*np.linalg.norm(v) + 1e-8))
            b = b + 0.06 * sim * v
        heart["usage"] += 1
        return b

    def use_hands(self, state) -> Tuple[str, float]:
        hands = self.organ_registry.get("hands")
        if not hands:
            return "think", 0.0
        logits = hands["selector"](state)
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        hands["usage"] += 1
        return hands["tools"][idx], float(probs[idx])

    def use_legs_plan(self, goal_state, steps=4) -> List[str]:
        legs = self.organ_registry.get("legs")
        if not legs:
            return ["think"] * steps
        h = legs["plan_proj"](goal_state)
        plan = []
        for _ in range(steps):
            h = np.tanh(legs["step_head"](h))
            s = h.sum()
            plan.append("act" if s > 0.3 else "think" if s > -0.3 else "recall")
        legs["plan"] = plan
        legs["usage"] += 1
        return plan

    def use_ears(self, token_ids: List[int]) -> np.ndarray:
        ears = self.organ_registry.get("ears")
        if not ears or not token_ids:
            return np.zeros(self.g.dim, np.float32)
        states = []
        for t, tid in enumerate(token_ids[-self.g.ctx:]):
            states.append(ears["embed"][tid] + ears["pos"][t % self.g.ctx])
        ears["usage"] += 1
        return 0.65 * states[-1] + 0.35 * np.mean(states, 0)

    def use_mouth(self, state, max_new=40, temperature=0.8) -> List[int]:
        mouth = self.organ_registry.get("mouth")
        if not mouth:
            return []
        ids = []
        h = state.copy()
        for _ in range(max_new):
            logit = mouth["head"](layer_norm(h)) / max(temperature, 1e-5)
            logit -= np.max(logit)
            p = np.exp(logit)
            p /= p.sum() + 1e-8
            nid = int(np.random.choice(len(p), p=p))
            ids.append(nid)
            h = 0.9 * h + 0.1 * rand_state(self.g.dim, 0.01)
        mouth["usage"] += 1
        return ids

    # -------------------------------------------------------------------------
    # Full cycle driven entirely by core-grown organs
    # -------------------------------------------------------------------------
    def cycle(self, text: str, tok) -> Dict:
        log = {"input": text, "organs_used": [], "generation": self.generation}

        ids = [tok["bos"]] + [tok["stoi"].get(c, tok["unk"]) for c in text]
        x = self.use_ears(ids)
        log["organs_used"].append("ears")

        # core thinks, biased by heart + memory (all core-grown)
        x = x + self.use_heart_bias(x)
        x = x + 0.2 * self.use_memory_read(x)
        x = self.forward(x)
        self.use_memory_write(x)
        log["organs_used"].extend(["heart", "memory", "brain"])

        plan = self.use_legs_plan(x)
        log["plan"] = plan
        log["organs_used"].append("legs")

        actions = []
        for step in plan:
            tool, conf = self.use_hands(x)
            actions.append({"step": step, "tool": tool, "conf": round(conf, 3)})
            x = self.forward(x)
        log["actions"] = actions
        log["organs_used"].append("hands")

        out_ids = self.use_mouth(x)
        log["utterance"] = "".join(
            tok["itos"][i] for i in out_ids
            if 0 <= i < len(tok["itos"]) and not tok["itos"][i].startswith("<")
        )
        log["organs_used"].append("mouth")
        return log

    def status(self):
        return {
            "generation": self.generation,
            "genome": asdict(self.g),
            "organs": {n: {"type": o["type"], "born": o["born"], "usage": o["usage"]}
                       for n, o in self.organ_registry.items()},
            "history_tail": self.history[-12:],
            "n_seeds": len(self.organ_seeds),
        }


# =============================================================================
# Bootstrap + Evolution Loop
# =============================================================================

REQUIRED = [
    ("brain", "generic"),       # core tissue itself is also registered
    ("memory", "memory"),
    ("ears", "ears"),
    ("mouth", "mouth"),
    ("hands", "hands"),
    ("heart", "heart"),
    ("legs", "legs"),
    ("attention", "attention"),
    ("buffer", "buffer"),
]

def build_tokenizer(vs=256):
    chars = [chr(i) for i in range(32, 127)]
    special = ["<pad>", "<unk>", "<bos>", "<eos>"]
    itos = (special + chars + [f"<e{i}>" for i in range(vs)])[:vs]
    stoi = {c: i for i, c in enumerate(itos)}
    return {"itos": itos, "stoi": stoi, "pad": 0, "unk": 1, "bos": 2, "eos": 3}

def evolve_loop(generations=5, cycles_per_gen=3):
    genome = CoreGenome()
    core = GeometricCore(genome)
    tok = build_tokenizer(genome.vocab_size)

    print("=" * 64)
    print("  METATRON CORE-EVOLVE")
    print("  The geometric core designs and evolves every organ")
    print("=" * 64)

    # Generation 0: core designs the entire body from itself
    core.design_body(REQUIRED)
    print("\n[gen 0] Core designed initial body:")
    for n, o in core.organ_registry.items():
        print(f"  - {n:12s} type={o['type']:10s} born=gen{o['born']}")

    prompts = [
        "what are you",
        "remember the flower of life",
        "plan how to learn",
        "speak about geometry",
    ]

    for gen in range(1, generations + 1):
        print(f"\n{'='*40} GENERATION {gen} {'='*40}")
        # Core mutates its own genome
        new_genome = core.mutate_genome()
        # Re-create core with new genome but keep organ seeds as heritage
        old_seeds = copy.deepcopy(core.organ_seeds)
        old_hist = core.history.copy()
        core = GeometricCore(new_genome)
        core.organ_seeds = old_seeds
        core.history = old_hist
        core.generation = gen

        # Core re-designs every organ from its new geometric state
        core.design_body(REQUIRED)

        # Core may invent one extra organ
        if np.random.rand() < 0.4:
            name = core.invent_organ(" emergent_function")
            print(f"  Core invented new organ: {name}")

        # Live cycles so organs get usage
        for i in range(cycles_per_gen):
            p = prompts[i % len(prompts)]
            log = core.cycle(p, tok)
            print(f"  cycle '{p[:30]}' → organs: {log['organs_used']}")
            print(f"           plan={log['plan']}  mouth='{log['utterance'][:40]}'")

        # Evolve the most-used organs harder
        usage = [(n, o["usage"]) for n, o in core.organ_registry.items()]
        usage.sort(key=lambda x: -x[1])
        for n, u in usage[:3]:
            if u > 0:
                core.evolve_organ(n)
                print(f"  evolved high-usage organ: {n} (usage={u})")

        st = core.status()
        print(f"  organs alive: {list(st['organs'].keys())}")

    print("\n" + "=" * 64)
    print("FINAL STATUS")
    print("=" * 64)
    st = core.status()
    print(json.dumps(st, indent=2, default=str))
    return core

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    evolve_loop(generations=args.gens, cycles_per_gen=args.cycles)
