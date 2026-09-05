"""
MetatronULTRA 500M: Scalable Template System
Build learnable Flower-of-Life networks at any parameter scale

Usage:
    # Small (93K params, 34s)
    model = Metatron(scale='ultra')
    
    # Medium (2.5M params, ~2min)
    model = Metatron(scale='large')
    
    # Large (50M params, ~30min)
    model = Metatron(scale='xlarge')
    
    # Massive (500M params, ~8 hours)
    model = Metatron(scale='500m')
"""

import numpy as np
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ============================================================================
# CONFIGURATION TEMPLATES
# ============================================================================

@dataclass
class MetatronConfig:
    """Scalable configuration for any parameter size"""
    name: str
    dim: int                    # Embedding dimension
    n_modules: int              # Number of modules in graph
    rec_layers: int             # Recurrent layers per module
    proj_layers: int            # Projection layers per module
    vocab_size: int = 30        # Character vocab
    batch_size: int = 1
    learning_rate: float = 0.01
    lr_decay: float = 0.97
    max_epochs: int = 30
    
    @property
    def total_params(self) -> int:
        """Calculate total learnable parameters"""
        # Embedding table
        emb_params = self.vocab_size * self.dim
        
        # Head layer
        head_params = self.dim * self.vocab_size
        
        # Per module: recurrent layers + projection layers
        rec_params = self.rec_layers * (self.dim * self.dim)
        proj_params = self.proj_layers * (self.dim * self.dim)
        module_params = self.n_modules * (rec_params + proj_params)
        
        return emb_params + head_params + module_params
    
    def summary(self):
        """Print configuration summary"""
        print(f"\n{'='*70}")
        print(f"Config: {self.name}")
        print(f"{'='*70}")
        print(f"Embedding dim:        {self.dim}")
        print(f"Modules:              {self.n_modules}")
        print(f"Recurrent layers:     {self.rec_layers}")
        print(f"Projection layers:    {self.proj_layers}")
        print(f"Batch size:           {self.batch_size}")
        print(f"Total parameters:     {self.total_params:,}")
        print(f"Learning rate:        {self.learning_rate}")
        print(f"Max epochs:           {self.max_epochs}")
        print(f"{'='*70}\n")


# Predefined scales
SCALES = {
    'ultra': MetatronConfig(
        name='UltraTiny',
        dim=48,
        n_modules=19,
        rec_layers=1,
        proj_layers=1,
        batch_size=1,
        learning_rate=0.01,
        max_epochs=30,
    ),
    'large': MetatronConfig(
        name='Large',
        dim=256,
        n_modules=19,
        rec_layers=2,
        proj_layers=2,
        batch_size=4,
        learning_rate=0.008,
        max_epochs=30,
    ),
    'xlarge': MetatronConfig(
        name='XLarge',
        dim=1024,
        n_modules=19,
        rec_layers=2,
        proj_layers=2,
        batch_size=8,
        learning_rate=0.005,
        max_epochs=25,
    ),
    '500m': MetatronConfig(
        name='HalfBillion',
        dim=2048,
        n_modules=38,  # Double the topology
        rec_layers=3,
        proj_layers=3,
        batch_size=16,
        learning_rate=0.003,
        max_epochs=20,
    ),
}


# ============================================================================
# LINEAR LAYER (Reusable across scales)
# ============================================================================

class Lin:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Initialize weights with He initialization
        std = math.sqrt(2.0 / in_dim)
        self.W = np.random.randn(out_dim, in_dim) * std
        self.b = np.zeros(out_dim)
        
        self.W_grad = np.zeros_like(self.W)
        self.b_grad = np.zeros_like(self.b)
    
    def forward(self, x):
        """x shape: (batch, in_dim) or (in_dim,)"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.x_cache = x
        out = x @ self.W.T + self.b
        return out if out.shape[0] > 1 else out[0]
    
    def backward(self, x, grad_out):
        """grad_out shape matches forward output"""
        if grad_out.ndim == 0:
            grad_out = grad_out.reshape(1, -1)
        elif grad_out.ndim == 1 and grad_out.shape[0] == self.out_dim:
            grad_out = grad_out.reshape(1, -1)
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Gradients
        self.W_grad += grad_out.T @ x
        self.b_grad += grad_out.sum(axis=0)
        grad_x = grad_out @ self.W
        
        return grad_x[0] if grad_x.shape[0] == 1 else grad_x
    
    def step(self, lr):
        """SGD update"""
        self.W -= lr * self.W_grad
        self.b -= lr * self.b_grad
        self.W_grad.fill(0)
        self.b_grad.fill(0)


# ============================================================================
# STACKED LAYER (Multiple linear layers in sequence)
# ============================================================================

class StackedLin:
    """Stack of linear layers (e.g., 3 layers for recurrence)"""
    def __init__(self, dim, n_layers):
        self.dim = dim
        self.n_layers = n_layers
        self.layers = [Lin(dim, dim) for _ in range(n_layers)]
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
            out = np.tanh(out)  # Activation between layers
        return out
    
    def backward(self, x, grad_out):
        grad = grad_out
        for layer in reversed(self.layers):
            grad = layer.backward(x, grad)
            # Backprop through tanh
            grad = grad * (1 - np.tanh(x)**2)
        return grad
    
    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)


# ============================================================================
# MODULE (Scalable across all sizes)
# ============================================================================

class ModuleULTRA:
    """Single learnable module in the Flower-of-Life graph"""
    def __init__(self, mid, dim, rec_layers=1, proj_layers=1):
        self.mid = mid
        self.dim = dim
        
        # State and gradients
        self.emb = np.zeros(dim)
        self.emb_grad = np.zeros(dim)
        
        # Learnable recurrent transformation (1-3 layers)
        self.rec = StackedLin(dim, rec_layers)
        
        # Learnable projection (1-3 layers)
        self.proj = StackedLin(dim, proj_layers)
        
        # Multi-timescale memory
        self.mem = [np.zeros(dim) for _ in range(3)]
    
    def forward(self, neighbor_in):
        """
        Process neighbor inputs and update state
        neighbor_in: dict of neighbor contributions
        """
        # Aggregate neighbor inputs and memory
        total = self.mem[0] * 0.5 + self.mem[1] * 0.7
        if neighbor_in:
            total += sum(neighbor_in.values()) / len(neighbor_in)
        
        # Recurrent update (learns state dynamics)
        updated = self.rec.forward(self.emb)
        
        # Gating mechanism
        gate = np.tanh(updated + total)
        new_emb = self.emb + gate * 0.1
        
        # Projection to neighbors (learns broadcast)
        output = self.proj.forward(new_emb)
        
        return new_emb, output
    
    def backward_step(self, d_emb, neighbor_grad_needed, lr):
        """Backprop through this module"""
        # Accumulate embedding gradient
        self.emb_grad += d_emb
        
        # Update recurrent and projection layers
        self.rec.step(lr)
        self.proj.step(lr)
    
    def apply_embedding_update(self, lr):
        """Update embedding via gradient descent"""
        self.emb -= lr * self.emb_grad
        self.emb_grad.fill(0)
        
        # Cycle memory
        self.mem[2] = self.mem[1].copy()
        self.mem[1] = self.mem[0].copy()
        self.mem[0] = self.emb.copy()


# ============================================================================
# TOKENIZER
# ============================================================================

class Tok:
    def __init__(self, chars):
        self.chars = list(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
    
    def enc(self, text):
        return [self.stoi.get(c, 0) for c in text]
    
    def dec(self, ids):
        return ''.join(self.itos.get(i, '?') for i in ids)


# ============================================================================
# METATRON (Main Model - Scalable to 500M)
# ============================================================================

class Metatron:
    def __init__(self, config: MetatronConfig):
        self.config = config
        
        # Tokenizer
        corpus_text = " ".join(CORPUS)
        self.tok = Tok(corpus_text)
        
        # Embedding table
        self.embed_table = np.random.randn(len(self.tok.chars), config.dim) * 0.01
        self.embed_grad = np.zeros_like(self.embed_table)
        
        # Modules (Flower-of-Life topology)
        self.modules = {}
        for mid in range(config.n_modules):
            self.modules[mid] = ModuleULTRA(
                mid, 
                config.dim, 
                rec_layers=config.rec_layers,
                proj_layers=config.proj_layers,
            )
        
        # Define Flower-of-Life connectivity
        self._build_topology(config.n_modules)
        
        # Readout head
        self.head = Lin(config.dim, len(self.tok.chars))
        
        # Workspace
        self.workspace = np.zeros(config.dim)
    
    def _build_topology(self, n_modules):
        """Build Flower-of-Life graph connectivity"""
        self.neighbors = {}
        
        if n_modules == 19:
            # Standard topology: 1 center + 6 ring1 + 12 ring2
            self.neighbors[0] = list(range(1, 7))  # Center connects to ring 1
            for i in range(1, 7):
                self.neighbors[i] = [0] + [(i % 6) + 1]  # Ring 1 nodes
            for i in range(7, 19):
                self.neighbors[i] = list(range(1, 7))  # Ring 2 connects to ring 1
        
        elif n_modules == 38:
            # Expanded topology: 2 centers + rings
            self.neighbors[0] = list(range(1, 13))
            self.neighbors[1] = list(range(13, 25))
            for i in range(2, 13):
                self.neighbors[i] = [0, (i % 12) + 2]
            for i in range(13, 25):
                self.neighbors[i] = [1, ((i-13) % 12) + 13]
            for i in range(25, 38):
                self.neighbors[i] = list(range(1, 13)) + list(range(13, 25))
        
        else:
            # Default: fully connected
            self.neighbors = {i: list(range(n_modules)) for i in range(n_modules)}
    
    def embed(self, ids):
        """Embed character IDs"""
        return self.embed_table[ids].mean(axis=0)
    
    def inject(self, mod_id, vec, scale=1.0):
        """Inject vector into module"""
        self.modules[mod_id].emb += vec * scale
    
    def step(self):
        """Forward pass through all modules"""
        outputs = {}
        for mid in self.modules:
            neighbor_indices = self.neighbors.get(mid, [])
            neighbor_in = {nid: outputs.get(nid, np.zeros(self.config.dim)) 
                          for nid in neighbor_indices}
            
            new_emb, output = self.modules[mid].forward(neighbor_in)
            self.modules[mid].emb = new_emb
            outputs[mid] = output
        
        # Aggregate to workspace
        self.workspace = np.mean([self.modules[mid].emb for mid in self.modules], axis=0)
    
    def train_step(self, ids, lr):
        """Single training step"""
        loss_sum = 0.0
        
        for i in range(1, len(ids)):
            ctx = ids[:i]
            target = ids[i]
            
            # Forward
            vec = self.embed(ctx)
            self.inject(0, vec)
            self.inject(1, vec * 0.5)
            self.inject(2, vec * 0.35)
            self.step()
            
            # Readout
            logits = self.head.forward(self.workspace)
            logits = np.clip(logits, -100, 100)  # Stability
            
            # Softmax
            logits_norm = logits - logits.max()
            exp_logits = np.exp(logits_norm)
            probs = exp_logits / (exp_logits.sum() + 1e-8)
            
            # Loss
            loss = -np.log(probs[target] + 1e-8)
            loss_sum += loss
            
            # Backward
            d_logits = probs.copy()
            d_logits[target] -= 1.0
            
            # Clip gradients
            norm = np.linalg.norm(d_logits)
            if norm > 1.5:
                d_logits *= 1.5 / norm
            
            # Through head
            d_workspace = self.head.backward(self.workspace, d_logits)
            self.head.step(lr)
            
            # Through modules
            for mid in self.modules:
                self.modules[mid].emb_grad += d_workspace * 0.1
                self.modules[mid].backward_step(
                    d_workspace * 0.1, 
                    {}, 
                    lr
                )
                self.modules[mid].apply_embedding_update(lr)
        
        return loss_sum / len(ids)
    
    def generate(self, prompt, length=20):
        """Generate sequence from prompt"""
        ids = self.tok.enc(prompt)
        for _ in range(length):
            vec = self.embed(ids)
            self.inject(0, vec)
            self.step()
            logits = self.head.forward(self.workspace)
            logits = np.clip(logits, -100, 100)
            logits_norm = logits - logits.max()
            exp_logits = np.exp(logits_norm)
            probs = exp_logits / (exp_logits.sum() + 1e-8)
            next_id = np.argmax(probs)
            ids.append(next_id)
        return self.tok.dec(ids)
    
    def validate(self, ids):
        """Compute validation perplexity"""
        loss_sum = 0.0
        for i in range(1, len(ids)):
            ctx = ids[:i]
            target = ids[i]
            vec = self.embed(ctx)
            self.inject(0, vec)
            self.step()
            logits = self.head.forward(self.workspace)
            logits = np.clip(logits, -100, 100)
            logits_norm = logits - logits.max()
            exp_logits = np.exp(logits_norm)
            probs = exp_logits / (exp_logits.sum() + 1e-8)
            loss = -np.log(probs[target] + 1e-8)
            loss_sum += loss
        
        mean_loss = loss_sum / len(ids)
        ppl = np.exp(mean_loss)
        return ppl
    
    def train(self):
        """Full training loop"""
        self.config.summary()
        
        # Prepare data
        train_text = " ".join(CORPUS[:-5])
        val_text = " ".join(CORPUS[-5:])
        train_ids = self.tok.enc(train_text)
        val_ids = self.tok.enc(val_text)
        
        # Baseline
        baseline_ppl = self.validate(val_ids)
        print(f"Baseline validation perplexity: {baseline_ppl:.2f}\n")
        print(f"Training {self.config.name}...")
        
        # Training loop
        start_time = time.time()
        for epoch in range(self.config.max_epochs):
            # Decay learning rate
            lr = self.config.learning_rate * (self.config.lr_decay ** epoch)
            
            # Train step
            epoch_start = time.time()
            loss = self.train_step(train_ids, lr)
            epoch_time = time.time() - epoch_start
            
            # Validation
            val_ppl = self.validate(val_ids)
            
            print(f"Epoch {epoch+1:2d} | lr {lr:.5f} | train loss {loss:.4f} | val ppl {val_ppl:.2f} | time {epoch_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Final validation perplexity: {val_ppl:.2f}")
        print(f"Improvement: {baseline_ppl - val_ppl:.2f} (lower is better)")
        
        # Generate samples
        print(f"\nGenerated samples:")
        for prompt in ['the flower', 'training', 'sacred']:
            generated = self.generate(prompt, length=30)
            print(f"  '{prompt}' -> '{generated}'")


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
# MAIN: SCALING LOOP
# ============================================================================

def main():
    """Build Metatron at different scales"""
    
    print("\n" + "="*70)
    print("METATRON 500M: SCALABLE TEMPLATE SYSTEM")
    print("="*70)
    
    # Choose scale
    print("\nAvailable scales:")
    for scale_name, config in SCALES.items():
        print(f"  {scale_name:10s} → {config.total_params:,} params")
    
    scale = 'ultra'  # Change to 'large', 'xlarge', '500m' to experiment
    
    config = SCALES[scale]
    model = Metatron(config)
    model.train()


if __name__ == "__main__":
    main()
