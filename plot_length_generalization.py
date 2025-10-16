#!/usr/bin/env python3
"""
Generate length generalization plots from partial experiment data.
Shows WITH vs WITHOUT state passing comparison as training progresses.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_summaries(output_dir):
    """Load summaries if they exist."""
    summary_path = Path(output_dir) / "all_summaries.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {summary_path}: {e}")
        return None

def plot_length_generalization(window_size):
    """Plot length generalization for a specific window size."""
    
    # Load data
    with_sp = load_summaries(f"length_gen_study_w{window_size}_500ckpt")
    without_sp = load_summaries(f"length_gen_study_w{window_size}_500ckpt_no_sp")
    
    if not with_sp:
        print(f"⚠️  No data yet for W={window_size} WITH state passing")
        return None
    
    has_without = without_sp is not None and len(without_sp) > 0
    
    # Create figure with subplots
    if has_without:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
    fig.suptitle(f'Length Generalization Study: W={window_size}\n' + 
                 ('WITH vs WITHOUT State Passing' if has_without else 'WITH State Passing (WITHOUT pending)'),
                 fontsize=16, fontweight='bold')
    
    # =========================================================================
    # Plot 1: Perplexity vs Sequence Length at different training checkpoints
    # =========================================================================
    ax1 = axes[0, 0]
    
    # Select MORE checkpoints to plot (show ALL for dense visualization)
    n_checkpoints = len(with_sp)
    # Instead of 5 checkpoints, show many more evenly spaced
    num_curves_to_show = min(100, n_checkpoints)  # Show up to 100 curves
    step_size = max(1, n_checkpoints // num_curves_to_show)
    plot_indices = list(range(0, n_checkpoints, step_size))
    if (n_checkpoints - 1) not in plot_indices:
        plot_indices.append(n_checkpoints - 1)  # Always include last
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_indices)))
    
    for idx, ckpt_idx in enumerate(plot_indices):
        ckpt = with_sp[ckpt_idx]
        step = ckpt['step']
        
        # Extract perplexity at different lengths
        seq_lengths = []
        ppls = []
        for mult in [1, 2, 5, 10, 20]:
            key = f'ttt_ppl_at_{mult}xW'
            if key in ckpt:
                seq_lengths.append(mult * window_size)
                ppls.append(ckpt[key])
        
        if seq_lengths:
            # Add label only for first and last to avoid legend clutter
            label = f'Step {step:,}' if idx in [0, len(plot_indices)-1] else None
            alpha = 0.3 if idx not in [0, len(plot_indices)-1] else 0.8  # Fade intermediate curves
            linewidth = 2 if idx in [0, len(plot_indices)-1] else 1
            markersize = 6 if idx in [0, len(plot_indices)-1] else 2
            
            ax1.plot(seq_lengths, ppls, 'o-', color=colors[idx], 
                    label=label, linewidth=linewidth, markersize=markersize, alpha=alpha)
    
    ax1.set_xlabel('Sequence Length (tokens)', fontsize=12)
    ax1.set_ylabel('Perplexity', fontsize=12)
    ax1.set_title('Perplexity vs Sequence Length During Training', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # =========================================================================
    # Plot 2: 10×W Improvement Over Training
    # =========================================================================
    ax2 = axes[0, 1]
    
    # WITH state passing
    steps_with = [s['step'] for s in with_sp]
    imp_10x_with = [s.get('improvement_at_10xW', 0) for s in with_sp]
    
    # Use smaller markers and thinner lines for dense data
    marker_size = 2 if len(with_sp) > 50 else 4
    line_width = 1 if len(with_sp) > 50 else 2
    
    ax2.plot(steps_with, imp_10x_with, 'o-', color='blue', 
            label=f'WITH state passing ({len(with_sp)} ckpts)', 
            linewidth=line_width, markersize=marker_size, alpha=0.7)
    
    # WITHOUT state passing (if available)
    if has_without:
        steps_without = [s['step'] for s in without_sp]
        imp_10x_without = [s.get('improvement_at_10xW', 0) for s in without_sp]
        
        marker_size_no_sp = 2 if len(without_sp) > 50 else 4
        line_width_no_sp = 1 if len(without_sp) > 50 else 2
        
        ax2.plot(steps_without, imp_10x_without, 's-', color='red', 
                label=f'WITHOUT state passing ({len(without_sp)} ckpts)', 
                linewidth=line_width_no_sp, markersize=marker_size_no_sp, alpha=0.7)
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('10×W Improvement (PPL)', fontsize=12)
    ax2.set_title(f'Length Generalization (10×W = {10*window_size} tokens)', 
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Add text annotation for latest values
    latest_with = imp_10x_with[-1]
    ax2.text(0.02, 0.98, f'Latest WITH: {latest_with:+.1f} PPL', 
            transform=ax2.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='blue', alpha=0.2))
    
    if has_without and imp_10x_without:
        latest_without = imp_10x_without[-1]
        ax2.text(0.02, 0.88, f'Latest WITHOUT: {latest_without:+.1f} PPL', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
    
    # =========================================================================
    # Plot 3: Training Loss Comparison
    # =========================================================================
    ax3 = axes[1, 0]
    
    # WITH state passing
    train_loss_with = [s.get('training_loss', 0) for s in with_sp 
                       if s.get('training_loss') is not None and s.get('training_loss') > 0]
    steps_with_loss = [with_sp[i]['step'] for i in range(len(with_sp)) 
                       if with_sp[i].get('training_loss') is not None and with_sp[i].get('training_loss') > 0]
    
    if train_loss_with:
        marker_size_loss = 1 if len(train_loss_with) > 50 else 3
        line_width_loss = 1 if len(train_loss_with) > 50 else 2
        
        ax3.plot(steps_with_loss, train_loss_with, 'o-', color='blue', 
                label='WITH state passing', linewidth=line_width_loss, 
                markersize=marker_size_loss, alpha=0.7)
    
    # WITHOUT state passing
    if has_without:
        train_loss_without = [s.get('training_loss', 0) for s in without_sp 
                              if s.get('training_loss') is not None and s.get('training_loss') > 0]
        steps_without_loss = [without_sp[i]['step'] for i in range(len(without_sp)) 
                              if without_sp[i].get('training_loss') is not None and without_sp[i].get('training_loss') > 0]
        
        if train_loss_without:
            marker_size_loss_no_sp = 1 if len(train_loss_without) > 50 else 3
            line_width_loss_no_sp = 1 if len(train_loss_without) > 50 else 2
            
            ax3.plot(steps_without_loss, train_loss_without, 's-', color='red', 
                    label='WITHOUT state passing', linewidth=line_width_loss_no_sp, 
                    markersize=marker_size_loss_no_sp, alpha=0.7)
    
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_ylabel('Training Loss', fontsize=12)
    ax3.set_title('Training Loss Progression', fontsize=13, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 4: Perplexity at Multiple Lengths (Latest Checkpoint)
    # =========================================================================
    ax4 = axes[1, 1]
    
    # Latest checkpoint WITH
    latest_with = with_sp[-1]
    
    lengths = []
    ppl_with = []
    ppl_sw_with = []
    
    for mult in [1, 2, 5, 10, 20]:
        ttt_key = f'ttt_ppl_at_{mult}xW'
        sw_key = f'sw_ppl_at_{mult}xW'
        if ttt_key in latest_with and sw_key in latest_with:
            lengths.append(mult * window_size)
            ppl_with.append(latest_with[ttt_key])
            ppl_sw_with.append(latest_with[sw_key])
    
    x = np.arange(len(lengths))
    width = 0.35
    
    if has_without and len(without_sp) > 0:
        latest_without = without_sp[-1]
        ppl_without = []
        ppl_sw_without = []
        
        for mult in [1, 2, 5, 10, 20]:
            ttt_key = f'ttt_ppl_at_{mult}xW'
            sw_key = f'sw_ppl_at_{mult}xW'
            if ttt_key in latest_without and sw_key in latest_without:
                ppl_without.append(latest_without[ttt_key])
                ppl_sw_without.append(latest_without[sw_key])
        
        # Plot grouped bars
        ax4.bar(x - width/2, ppl_with, width, label='WITH SP (TTT)', color='blue', alpha=0.7)
        if ppl_without and len(ppl_without) == len(x):
            ax4.bar(x + width/2, ppl_without, width, label='WITHOUT SP (TTT)', color='red', alpha=0.7)
        
        # Add sliding window baseline
        ax4.plot(x, ppl_sw_with, 'k--', label='SW baseline', linewidth=2, marker='x', markersize=8)
    else:
        # Only WITH available
        ax4.bar(x, ppl_with, width, label='WITH SP (TTT)', color='blue', alpha=0.7)
        ax4.plot(x, ppl_sw_with, 'k--', label='SW baseline', linewidth=2, marker='x', markersize=8)
    
    ax4.set_xlabel('Sequence Length', fontsize=12)
    ax4.set_ylabel('Perplexity', fontsize=12)
    ax4.set_title(f'Latest Checkpoint Comparison (Step {latest_with["step"]:,})', 
                 fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{l}' for l in lengths])
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("plots_length_gen")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"length_gen_w{window_size}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()
    
    return {
        'window_size': window_size,
        'with_checkpoints': len(with_sp),
        'without_checkpoints': len(without_sp) if has_without else 0,
        'latest_step_with': with_sp[-1]['step'],
        'latest_10xW_with': with_sp[-1].get('improvement_at_10xW', 0),
        'latest_10xW_without': without_sp[-1].get('improvement_at_10xW', 0) if has_without and without_sp else None,
    }


def create_summary_plot(window_sizes=[16, 32, 64]):
    """Create a summary comparison plot across all window sizes."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Length Generalization Summary: WITH vs WITHOUT State Passing', 
                 fontsize=16, fontweight='bold')
    
    for idx, W in enumerate(window_sizes):
        ax = axes[idx]
        
        with_sp = load_summaries(f"length_gen_study_w{W}_500ckpt")
        without_sp = load_summaries(f"length_gen_study_w{W}_500ckpt_no_sp")
        
        if not with_sp:
            ax.text(0.5, 0.5, f'W={W}\nNo data yet', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'W={W}', fontsize=13, fontweight='bold')
            continue
        
        # Plot 10×W improvement over training
        steps_with = [s['step'] for s in with_sp]
        imp_with = [s.get('improvement_at_10xW', 0) for s in with_sp]
        
        # Adaptive marker/line size based on number of checkpoints
        marker_size_summary = 1 if len(with_sp) > 50 else 3
        line_width_summary = 1 if len(with_sp) > 50 else 2
        
        ax.plot(steps_with, imp_with, 'o-', color='blue', 
               label=f'WITH ({len(with_sp)} ckpts)', 
               linewidth=line_width_summary, markersize=marker_size_summary, alpha=0.7)
        
        if without_sp and len(without_sp) > 0:
            steps_without = [s['step'] for s in without_sp]
            imp_without = [s.get('improvement_at_10xW', 0) for s in without_sp]
            
            marker_size_no_sp_summary = 1 if len(without_sp) > 50 else 3
            line_width_no_sp_summary = 1 if len(without_sp) > 50 else 2
            
            ax.plot(steps_without, imp_without, 's-', color='red', 
                   label=f'WITHOUT ({len(without_sp)} ckpts)', 
                   linewidth=line_width_no_sp_summary, markersize=marker_size_no_sp_summary, alpha=0.7)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel('10×W Improvement (PPL)', fontsize=11)
        ax.set_title(f'W={W} (10×W = {10*W} tokens)', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add latest value annotation
        latest_with = imp_with[-1]
        color_with = 'green' if latest_with < 0 else 'orange'
        ax.text(0.98, 0.02, f'Latest WITH:\n{latest_with:+.0f} PPL', 
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor=color_with, alpha=0.3),
               fontsize=10)
        
        if without_sp and len(without_sp) > 0:
            latest_without = imp_without[-1]
            color_without = 'green' if latest_without < 0 else 'orange'
            ax.text(0.98, 0.15, f'Latest WITHOUT:\n{latest_without:+.0f} PPL', 
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor=color_without, alpha=0.3),
                   fontsize=10)
    
    plt.tight_layout()
    
    output_dir = Path("plots_length_gen")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "summary_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()


def main():
    print("=" * 70)
    print("LENGTH GENERALIZATION PLOTS")
    print("=" * 70)
    print()
    
    results = []
    for W in [16, 32, 64]:
        print(f"\n📊 Generating plots for W={W}...")
        result = plot_length_generalization(W)
        if result:
            results.append(result)
    
    print(f"\n📊 Generating summary comparison plot...")
    create_summary_plot([16, 32, 64])
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for r in results:
        W = r['window_size']
        print(f"\nW={W}:")
        print(f"  WITH state passing: {r['with_checkpoints']}/500 checkpoints (step {r['latest_step_with']:,})")
        print(f"  WITHOUT state passing: {r['without_checkpoints']}/500 checkpoints")
        print(f"  Latest 10×W improvement:")
        print(f"    WITH:    {r['latest_10xW_with']:+.1f} PPL")
        if r['latest_10xW_without'] is not None:
            print(f"    WITHOUT: {r['latest_10xW_without']:+.1f} PPL")
            diff = r['latest_10xW_without'] - r['latest_10xW_with']
            better = "WITHOUT" if diff < 0 else "WITH"
            print(f"    → {better} is better by {abs(diff):.1f} PPL")
        else:
            print(f"    WITHOUT: (not started yet)")
    
    print("\n" + "=" * 70)
    print("📁 Plots saved to: plots_length_gen/")
    print("=" * 70)
    
    # List files
    output_dir = Path("plots_length_gen")
    if output_dir.exists():
        print("\nGenerated files:")
        for f in sorted(output_dir.glob("*.png")):
            size_kb = f.stat().st_size / 1024
            print(f"  • {f.name} ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
