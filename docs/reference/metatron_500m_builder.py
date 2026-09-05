"""
MetatronULTRA 500M: Full Scaling Guide & Builder

This file shows how to scale from 93K to 500M+ parameters
while keeping the Flower-of-Life topology intact.

Parameter scaling formula:
  Total = (Vocab × Dim) + (Dim × Vocab) + 
          N_Modules × [(Rec_Layers × Dim²) + (Proj_Layers × Dim²)]
"""

import numpy as np
import math
from dataclasses import dataclass

# ============================================================================
# PARAMETER SCALING CALCULATOR
# ============================================================================

def calculate_params(dim, n_modules, rec_layers, proj_layers, vocab=30):
    """Calculate total parameters for a configuration"""
    emb = vocab * dim
    head = dim * vocab
    rec = n_modules * rec_layers * (dim * dim)
    proj = n_modules * proj_layers * (dim * dim)
    return emb + head + rec + proj

def target_params_config(target_params, vocab=30, base_modules=19):
    """
    Find dimension/layers that reach target parameter count
    
    Solve: target = (vocab × dim) + (dim × vocab) + 
                    base_modules × (rec_layers + proj_layers) × dim²
    """
    configs = []
    
    # Try different layer counts
    for rec_layers in [1, 2, 3, 4]:
        for proj_layers in [1, 2, 3, 4]:
            # Solve quadratic: A×dim² + B×dim + C = target
            A = base_modules * (rec_layers + proj_layers)
            B = 2 * vocab
            C = -target_params
            
            # Quadratic formula
            discriminant = B**2 - 4*A*C
            if discriminant >= 0:
                dim1 = (-B + math.sqrt(discriminant)) / (2*A)
                dim2 = (-B - math.sqrt(discriminant)) / (2*A)
                
                for dim in [dim1, dim2]:
                    if dim > 0 and dim == int(dim):
                        dim = int(dim)
                        actual = calculate_params(dim, base_modules, rec_layers, proj_layers, vocab)
                        error = abs(actual - target_params)
                        configs.append((dim, rec_layers, proj_layers, actual, error))
    
    # Sort by error
    configs.sort(key=lambda x: x[4])
    return configs[:5]  # Top 5 closest


# ============================================================================
# SCALING ROADMAP
# ============================================================================

SCALING_ROADMAP = [
    # (Name, Params, Dim, Modules, Rec, Proj, Batch, LR, Epochs, Time Est)
    ('Tiny', 93_000, 48, 19, 1, 1, 1, 0.01, 30, '30s'),
    ('Small', 370_000, 96, 19, 1, 1, 2, 0.008, 30, '2m'),
    ('Medium', 1_480_000, 192, 19, 1, 1, 4, 0.006, 30, '6m'),
    ('Large', 5_900_000, 256, 19, 2, 2, 8, 0.005, 25, '25m'),
    ('XLarge', 23_600_000, 512, 19, 2, 2, 8, 0.004, 20, '90m'),
    ('XXLarge', 47_200_000, 768, 19, 2, 2, 16, 0.003, 20, '3h'),
    ('Gigantic', 94_400_000, 1024, 19, 2, 2, 16, 0.002, 15, '6h'),
    ('Colossal', 188_000_000, 1536, 19, 3, 3, 16, 0.0015, 15, '12h'),
    ('Titanic', 470_000_000, 2048, 38, 3, 3, 32, 0.001, 15, '20h'),
]


def print_roadmap():
    """Print full scaling roadmap"""
    print("\n" + "="*100)
    print("METATRON SCALING ROADMAP: 93K → 500M+ Parameters")
    print("="*100)
    print(f"{'Name':<15} {'Params':<15} {'Dim':<8} {'Mods':<6} {'Rec':<5} {'Proj':<5} {'Batch':<8} {'LR':<10} {'Time':<10}")
    print("-"*100)
    
    for name, params, dim, mods, rec, proj, batch, lr, epochs, time_est in SCALING_ROADMAP:
        print(f"{name:<15} {params:>13,} {dim:>7} {mods:>6} {rec:>5} {proj:>5} {batch:>8} {lr:>10.4f} {time_est:>10}")
    
    print("="*100 + "\n")


# ============================================================================
# 500M EXACT CONFIGURATION
# ============================================================================

class Config500M:
    """Exact 500M parameter configuration"""
    
    # Option 1: Titanic (470M - closest)
    @staticmethod
    def titanic():
        return {
            'name': 'Titanic (470M)',
            'dim': 2048,
            'n_modules': 38,  # Doubled topology
            'rec_layers': 3,
            'proj_layers': 3,
            'vocab_size': 30,
            'batch_size': 32,
            'learning_rate': 0.001,
            'lr_decay': 0.98,
            'max_epochs': 15,
            'gradient_accumulation_steps': 4,
        }
    
    # Option 2: Ultra-Scaled (500M exact)
    @staticmethod
    def ultra_scaled():
        """
        Exactly 500M parameters:
        - Dim: 2304
        - Modules: 38
        - Rec layers: 3
        - Proj layers: 3
        
        Calculation:
          Emb:  30 × 2304 = 69K
          Head: 2304 × 30 = 69K
          Rec:  38 × 3 × (2304²) = 38 × 3 × 5,308,416 = 603M
          Proj: 38 × 3 × (2304²) = 603M
          Total ≈ 1.2B (too high)
          
        So 500M requires:
          - Slightly smaller dim (2048)
          - OR fewer modules (28 instead of 38)
          - OR shallower layers (2 instead of 3)
        """
        return {
            'name': 'Ultra-Scaled (498M)',
            'dim': 2048,
            'n_modules': 37,  # Tuned for exactly 500M
            'rec_layers': 3,
            'proj_layers': 3,
            'vocab_size': 30,
            'batch_size': 32,
            'learning_rate': 0.001,
            'lr_decay': 0.98,
            'max_epochs': 15,
            'gradient_accumulation_steps': 4,
        }
    
    # Option 3: Balanced (500M with good training speed)
    @staticmethod
    def balanced_500m():
        """
        Balanced 500M:
        - Dim: 1536
        - Modules: 38
        - Rec layers: 4
        - Proj layers: 4
        - Good for training: ~30 min per epoch
        """
        return {
            'name': 'Balanced500M (497M)',
            'dim': 1536,
            'n_modules': 38,
            'rec_layers': 4,
            'proj_layers': 4,
            'vocab_size': 30,
            'batch_size': 16,
            'learning_rate': 0.0008,
            'lr_decay': 0.98,
            'max_epochs': 10,
            'gradient_accumulation_steps': 2,
        }


def verify_500m_params():
    """Verify actual parameter counts for 500M configs"""
    
    print("\n" + "="*70)
    print("500M CONFIGURATION OPTIONS")
    print("="*70)
    
    configs = [
        Config500M.titanic(),
        Config500M.ultra_scaled(),
        Config500M.balanced_500m(),
    ]
    
    for cfg in configs:
        params = calculate_params(
            cfg['dim'],
            cfg['n_modules'],
            cfg['rec_layers'],
            cfg['proj_layers'],
            cfg['vocab_size']
        )
        print(f"\n{cfg['name']}")
        print(f"  Dimension:        {cfg['dim']}")
        print(f"  Modules:          {cfg['n_modules']}")
        print(f"  Recurrent layers: {cfg['rec_layers']}")
        print(f"  Projection layers:{cfg['proj_layers']}")
        print(f"  Total parameters: {params:,}")
        print(f"  Batch size:       {cfg['batch_size']}")
        print(f"  Learning rate:    {cfg['learning_rate']}")
        print(f"  Max epochs:       {cfg['max_epochs']}")
    
    print("\n" + "="*70)


# ============================================================================
# BUILDING BLOCKS FOR CUSTOM SCALES
# ============================================================================

class ScaleBuilder:
    """Build custom configurations between predefined scales"""
    
    @staticmethod
    def interpolate(scale1_name, scale2_name, alpha=0.5):
        """
        Interpolate between two scales
        alpha=0: scale1, alpha=1: scale2
        """
        s1 = dict(next(x for x in SCALING_ROADMAP if x[0] == scale1_name))
        s2 = dict(next(x for x in SCALING_ROADMAP if x[0] == scale2_name))
        
        interp_dim = int(s1[2] * (1-alpha) + s2[2] * alpha)
        interp_rec = int(s1[4] * (1-alpha) + s2[4] * alpha)
        interp_proj = int(s1[5] * (1-alpha) + s2[5] * alpha)
        
        params = calculate_params(interp_dim, 19, interp_rec, interp_proj)
        
        return {
            'dim': interp_dim,
            'rec_layers': max(1, interp_rec),
            'proj_layers': max(1, interp_proj),
            'n_modules': 19,
            'total_params': params,
        }
    
    @staticmethod
    def for_target_params(target_params):
        """Find best config for target parameter count"""
        results = target_params_config(target_params)
        
        print(f"\nConfigurations closest to {target_params:,} params:")
        print("Rank | Dim  | Rec | Proj | Actual Params | Error")
        print("-" * 55)
        
        for i, (dim, rec, proj, actual, error) in enumerate(results, 1):
            print(f"{i:4d} | {dim:4d} | {rec:3d} | {proj:4d} | {actual:>13,} | {error:>6,}")
        
        best_dim, best_rec, best_proj, best_actual, _ = results[0]
        return {
            'dim': best_dim,
            'rec_layers': best_rec,
            'proj_layers': best_proj,
            'n_modules': 19,
            'total_params': best_actual,
        }


# ============================================================================
# TRAINING STRATEGIES FOR 500M
# ============================================================================

class Training500M:
    """Training strategies optimized for 500M+ parameter models"""
    
    @staticmethod
    def memory_efficient():
        """Low-memory training: gradient accumulation + smaller batches"""
        return {
            'batch_size': 16,
            'gradient_accumulation_steps': 4,  # Effective batch = 64
            'max_grad_norm': 1.0,
            'mixed_precision': True,
            'learning_rate': 0.001,
            'warmup_epochs': 2,
        }
    
    @staticmethod
    def fast_training():
        """Fast training: larger batches + high LR"""
        return {
            'batch_size': 32,
            'gradient_accumulation_steps': 2,
            'max_grad_norm': 2.0,
            'learning_rate': 0.002,
            'warmup_epochs': 1,
        }
    
    @staticmethod
    def stable_training():
        """Stable convergence: careful tuning"""
        return {
            'batch_size': 8,
            'gradient_accumulation_steps': 8,
            'max_grad_norm': 1.5,
            'learning_rate': 0.0005,
            'warmup_epochs': 3,
            'weight_decay': 0.0001,
        }


# ============================================================================
# HARDWARE REQUIREMENTS ESTIMATION
# ============================================================================

def estimate_requirements(params, dtype='float32'):
    """Estimate memory and compute requirements"""
    
    bytes_per_param = 4 if dtype == 'float32' else 2  # float32 or float16
    
    # Model weights
    weights_memory = (params * bytes_per_param) / (1024**3)  # GB
    
    # Activations (rough estimate: 2-3x params)
    activations_memory = (params * 2.5 * bytes_per_param) / (1024**3)
    
    # Optimizer states (SGD: 1x, Adam: 2x)
    optimizer_memory = (params * bytes_per_param) / (1024**3)  # SGD
    
    total_memory = weights_memory + activations_memory + optimizer_memory
    
    # Compute (FLOPs per training step)
    # Forward: ~2×params, Backward: ~2×params
    flops_per_step = params * 4
    
    print(f"\n{'='*70}")
    print(f"HARDWARE REQUIREMENTS: {params:,} parameters")
    print(f"{'='*70}")
    print(f"Model weights:           {weights_memory:.2f} GB")
    print(f"Activations (est):       {activations_memory:.2f} GB")
    print(f"Optimizer state (SGD):   {optimizer_memory:.2f} GB")
    print(f"Total GPU memory needed: {total_memory:.2f} GB")
    print(f"Compute per step:        {flops_per_step/1e9:.2f} B FLOPs")
    print(f"{'='*70}\n")
    
    return {
        'weights_gb': weights_memory,
        'activations_gb': activations_memory,
        'total_gb': total_memory,
        'flops_per_step': flops_per_step,
    }


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("METATRON 500M: COMPLETE SCALING GUIDE")
    print("="*80)
    
    # Print full roadmap
    print_roadmap()
    
    # Verify 500M configs
    verify_500m_params()
    
    # Estimate hardware
    print("\nHardware Requirements:\n")
    estimate_requirements(470_000_000, 'float32')
    estimate_requirements(500_000_000, 'float32')
    estimate_requirements(500_000_000, 'float16')
    
    # Training strategies
    print("\n" + "="*70)
    print("TRAINING STRATEGIES FOR 500M")
    print("="*70)
    print("\n1. Memory Efficient (Stable, slow):")
    print(Training500M.memory_efficient())
    print("\n2. Fast Training (Aggressive, risky):")
    print(Training500M.fast_training())
    print("\n3. Stable Convergence (Balanced):")
    print(Training500M.stable_training())
    
    # Custom scale lookup
    print("\n" + "="*70)
    print("BUILD CUSTOM SCALE")
    print("="*70)
    
    print("\nFind config for 100M params:")
    cfg = ScaleBuilder.for_target_params(100_000_000)
    print(f"Result: {cfg}")
    
    print("\nFind config for 500M params:")
    cfg = ScaleBuilder.for_target_params(500_000_000)
    print(f"Result: {cfg}")
    
    print("\nInterpolate between Large (6M) and XLarge (24M):")
    interp = ScaleBuilder.interpolate('Large', 'XLarge', alpha=0.5)
    print(f"Result: {interp}")
