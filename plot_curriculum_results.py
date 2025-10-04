#!/usr/bin/env python3
"""
Plot curriculum results: smoothed perplexity evolution across positions.

Reads checkpoint JSONs from a curriculum run and generates a plot showing
TTT and SW perplexity with a moving average (window = W/4) from W to 20×W
as training progresses.

Usage:
  python plot_curriculum_results.py --curriculum_dir curriculum_exp
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--curriculum_dir", type=str, required=True, help="Path to curriculum experiment directory")
    p.add_argument("--output", type=str, default=None, help="Output directory (default: same as curriculum_dir)")
    p.add_argument("--window_size", type=int, default=None, help="Smoothing window size (default: W/4)")
    return p.parse_args()


def moving_average(arr, window_size):
    """Compute moving average with given window size."""
    if window_size <= 1:
        return arr
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')


def main():
    args = parse_args()
    
    curriculum_dir = Path(args.curriculum_dir)
    if not curriculum_dir.exists():
        raise FileNotFoundError(f"Curriculum directory not found: {curriculum_dir}")
    
    out_dir = Path(args.output) if args.output else curriculum_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    meta_path = curriculum_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {curriculum_dir}")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    W = meta["training_window"]
    T = meta["eval_max_length"]
    
    # Default smoothing window: W/4
    smooth_window = args.window_size if args.window_size else max(1, W // 4)
    
    print(f"Loading curriculum results from {curriculum_dir}")
    print(f"  Training window (W): {W}")
    print(f"  Eval max length (T): {T}")
    print(f"  Smoothing window: {smooth_window}")
    
    # Find all checkpoint directories
    checkpoint_dirs = sorted(curriculum_dir.glob("checkpoint_*"))
    if not checkpoint_dirs:
        raise FileNotFoundError("No checkpoint directories found")
    
    print(f"  Found {len(checkpoint_dirs)} checkpoints")
    
    # Load all checkpoint data from summary.json files
    checkpoint_data = []
    for ckpt_dir in checkpoint_dirs:
        # Load the summary JSON
        summary_path = ckpt_dir / "summary.json"
        if not summary_path.exists():
            print(f"  Warning: No summary.json in {ckpt_dir.name}, skipping")
            continue
        
        with open(summary_path, "r") as f:
            data = json.load(f)
        
        # Extract checkpoint index from directory name
        ckpt_idx = int(ckpt_dir.name.split("_")[-1])
        
        # Extract key positions (1xW, 2xW, 5xW, 10xW, 20xW)
        checkpoint_data.append({
            "checkpoint_idx": ckpt_idx,
            "step": data.get("step", ckpt_idx),
            "x_positions": [
                data["x_pos_at_1xW"],
                data["x_pos_at_2xW"],
                data["x_pos_at_5xW"],
                data["x_pos_at_10xW"],
                data["x_pos_at_20xW"],
            ],
            "ttt_ppls": [
                data["ttt_ppl_at_1xW"],
                data["ttt_ppl_at_2xW"],
                data["ttt_ppl_at_5xW"],
                data["ttt_ppl_at_10xW"],
                data["ttt_ppl_at_20xW"],
            ],
            "sw_ppls": [
                data["sw_ppl_at_1xW"],
                data["sw_ppl_at_2xW"],
                data["sw_ppl_at_5xW"],
                data["sw_ppl_at_10xW"],
                data["sw_ppl_at_20xW"],
            ],
            "training_window": data["training_window"],
        })
    
    if not checkpoint_data:
        raise ValueError("No valid checkpoint data loaded")
    
    # Sort by checkpoint index
    checkpoint_data.sort(key=lambda x: x["checkpoint_idx"])
    
    print(f"  Loaded {len(checkpoint_data)} valid checkpoints")
    
    # Create the plot: lines connecting key positions across training
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Length Generalization Evolution During Training (W={W})", 
                 fontsize=14, fontweight='bold')
    
    # Use a colormap for progression through training
    cmap = plt.cm.viridis
    n_checkpoints = len(checkpoint_data)
    
    # Plot each checkpoint as a line connecting the key positions
    for i, ckpt in enumerate(checkpoint_data):
        x_positions = np.array(ckpt["x_positions"])
        ttt = np.array(ckpt["ttt_ppls"])
        sw = np.array(ckpt["sw_ppls"])
        
        # Color based on training progress
        color = cmap(i / (n_checkpoints - 1))
        alpha = 0.3 + 0.7 * (i / (n_checkpoints - 1))  # Fade in as training progresses
        
        # Show label for some checkpoints
        show_label = (i == 0 or i == n_checkpoints - 1 or i % max(1, n_checkpoints // 5) == 0)
        label = f"Step {ckpt['step']}" if show_label else ""
        
        # Plot TTT
        ax1.plot(x_positions, ttt, color=color, alpha=alpha, linewidth=2, marker='o', markersize=4,
                label=label)
        
        # Plot SW
        ax2.plot(x_positions, sw, color=color, alpha=alpha, linewidth=2, marker='o', markersize=4,
                label=label)
    
    # Mark key positions with vertical lines
    for mult, label in [(1, '1×W'), (2, '2×W'), (5, '5×W'), (10, '10×W'), (20, '20×W')]:
        pos = W * mult
        ax1.axvline(pos, color='red', linestyle=':', alpha=0.3, linewidth=0.8)
        ax2.axvline(pos, color='red', linestyle=':', alpha=0.3, linewidth=0.8)
        # Add label at top
        ax1.text(pos, ax1.get_ylim()[1] * 0.98, label, ha='center', va='top', 
                fontsize=8, color='red', alpha=0.7)

    
    # Formatting
    ax1.set_ylabel("Perplexity (TTT)")
    ax1.set_title("TTT: Full-Context")
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel("Perplexity (SW)")
    ax2.set_xlabel("Position (tokens)")
    ax2.set_title(f"SW: Sliding Window (context ≤ {W})")
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar to show training progress
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=checkpoint_data[-1]["step"]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label('Training Step', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    plot_path = out_dir / "curriculum_length_gen_evolution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved plot: {plot_path}")
    plt.close()
    
    # Generate summary statistics
    print("\n📊 Summary Statistics:")
    print(f"  Checkpoints: {len(checkpoint_data)}")
    print(f"  Step range: {checkpoint_data[0]['step']} → {checkpoint_data[-1]['step']}")
    
    # Compare first vs last at key positions
    first_ckpt = checkpoint_data[0]
    last_ckpt = checkpoint_data[-1]
    
    print(f"\n  @1×W (position {W}):")
    print(f"    Initial:  TTT={first_ckpt['ttt_ppls'][0]:.1f}, SW={first_ckpt['sw_ppls'][0]:.1f}")
    print(f"    Final:    TTT={last_ckpt['ttt_ppls'][0]:.1f}, SW={last_ckpt['sw_ppls'][0]:.1f}")
    print(f"    TTT improvement: {first_ckpt['ttt_ppls'][0]/last_ckpt['ttt_ppls'][0]:.2f}×")
    
    print(f"\n  @10×W (position {10*W}):")
    print(f"    Initial:  TTT={first_ckpt['ttt_ppls'][3]:.1f}, SW={first_ckpt['sw_ppls'][3]:.1f}, Ratio={first_ckpt['sw_ppls'][3]/first_ckpt['ttt_ppls'][3]:.3f}")
    print(f"    Final:    TTT={last_ckpt['ttt_ppls'][3]:.1f}, SW={last_ckpt['sw_ppls'][3]:.1f}, Ratio={last_ckpt['sw_ppls'][3]/last_ckpt['ttt_ppls'][3]:.3f}")
    print(f"    TTT improvement: {first_ckpt['ttt_ppls'][3]/last_ckpt['ttt_ppls'][3]:.2f}×")
    print(f"    SW improvement: {first_ckpt['sw_ppls'][3]/last_ckpt['sw_ppls'][3]:.2f}×")
    
    print(f"\n  @20×W (position {20*W}):")
    print(f"    Initial:  TTT={first_ckpt['ttt_ppls'][4]:.1f}, SW={first_ckpt['sw_ppls'][4]:.1f}")
    print(f"    Final:    TTT={last_ckpt['ttt_ppls'][4]:.1f}, SW={last_ckpt['sw_ppls'][4]:.1f}")
    print(f"    TTT advantage: {(last_ckpt['sw_ppls'][4] - last_ckpt['ttt_ppls'][4]):.1f} PPL")



if __name__ == "__main__":
    main()
