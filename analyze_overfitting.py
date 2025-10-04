"""
Analyze overfitting study results for W=64 and W=128.
Compares full 100k-step training curves to identify overfitting patterns.
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_experiment_results(exp_dir):
    """Load all checkpoint results from an experiment directory."""
    checkpoints = sorted(glob.glob(str(Path(exp_dir) / "checkpoint_*/summary.json")))
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {exp_dir}")
    
    steps = []
    ttt_1xW = []
    sw_1xW = []
    ttt_10xW = []
    sw_10xW = []
    ttt_20xW = []
    sw_20xW = []
    
    for ckpt_path in checkpoints:
        with open(ckpt_path) as f:
            data = json.load(f)
        steps.append(data["step"])
        ttt_1xW.append(data["ttt_ppl_at_1xW"])
        sw_1xW.append(data["sw_ppl_at_1xW"])
        ttt_10xW.append(data.get("ttt_ppl_at_10xW", None))
        sw_10xW.append(data.get("sw_ppl_at_10xW", None))
        ttt_20xW.append(data.get("ttt_ppl_at_20xW", None))
        sw_20xW.append(data.get("sw_ppl_at_20xW", None))
    
    return {
        'steps': steps,
        'ttt_1xW': ttt_1xW,
        'sw_1xW': sw_1xW,
        'ttt_10xW': ttt_10xW,
        'sw_10xW': sw_10xW,
        'ttt_20xW': ttt_20xW,
        'sw_20xW': sw_20xW,
    }


def detect_overfitting(steps, ppl_values, window=5):
    """
    Detect overfitting by finding when PPL starts increasing consistently.
    Returns: (overfitting_detected, overfitting_step, best_step, best_ppl)
    """
    if len(ppl_values) < window + 1:
        return False, None, steps[np.argmin(ppl_values)], min(ppl_values)
    
    best_idx = np.argmin(ppl_values)
    best_step = steps[best_idx]
    best_ppl = ppl_values[best_idx]
    
    # Check if PPL increases consistently after best point
    if best_idx < len(ppl_values) - window:
        # Count how many subsequent points are worse
        worse_count = sum(1 for i in range(best_idx + 1, len(ppl_values)) 
                         if ppl_values[i] > best_ppl * 1.02)  # 2% tolerance
        
        if worse_count >= window:
            return True, best_step, best_step, best_ppl
    
    return False, None, best_step, best_ppl


def main():
    print("\n" + "=" * 80)
    print("TTT OVERFITTING STUDY ANALYSIS")
    print("=" * 80)
    
    # Load results
    experiments = {}
    for exp_name, exp_dir in [("W=64", "overfitting_w64"), ("W=128", "overfitting_w128")]:
        if Path(exp_dir).exists():
            try:
                experiments[exp_name] = load_experiment_results(exp_dir)
                print(f"✓ Loaded {exp_name}: {len(experiments[exp_name]['steps'])} checkpoints")
            except Exception as e:
                print(f"✗ Failed to load {exp_name}: {e}")
        else:
            print(f"✗ Directory not found: {exp_dir}")
    
    if not experiments:
        print("\n❌ No experiment results found!")
        return
    
    # Create comprehensive analysis plot
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = {"W=64": "red", "W=128": "blue"}
    
    # Plot 1: Training window PPL (1×W) - Overfitting detection
    ax1 = fig.add_subplot(gs[0, :2])
    for exp_name, data in experiments.items():
        steps = data['steps']
        ttt_1xW = data['ttt_1xW']
        
        # Detect overfitting
        overfitting, overfit_step, best_step, best_ppl = detect_overfitting(steps, ttt_1xW)
        
        ax1.plot(steps, ttt_1xW, '-', color=colors[exp_name], linewidth=2, 
                label=f'{exp_name} TTT@1×W', alpha=0.8)
        ax1.plot(steps, data['sw_1xW'], '--', color=colors[exp_name], linewidth=2,
                label=f'{exp_name} SW@1×W', alpha=0.5)
        
        # Mark best point
        ax1.scatter([best_step], [best_ppl], color=colors[exp_name], s=100, 
                   marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        
        # Mark overfitting region
        if overfitting:
            ax1.axvline(overfit_step, color=colors[exp_name], linestyle=':', alpha=0.5)
            ax1.text(overfit_step, best_ppl * 0.9, f'{exp_name}\nOverfit at\nstep {overfit_step:,}',
                    ha='center', fontsize=9, color=colors[exp_name],
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax1.text(best_step, best_ppl * 0.9, f'{exp_name}\nBest: step {best_step:,}\nPPL {best_ppl:.1f}',
                    ha='center', fontsize=9, color=colors[exp_name],
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perplexity (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Window Perplexity (1×W) - Overfitting Detection', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Long sequence PPL (10×W)
    ax2 = fig.add_subplot(gs[1, :2])
    for exp_name, data in experiments.items():
        if data['ttt_10xW'][0] is not None:
            ax2.plot(data['steps'], data['ttt_10xW'], '-', color=colors[exp_name], 
                    linewidth=2, label=f'{exp_name} TTT@10×W', alpha=0.8)
            ax2.plot(data['steps'], data['sw_10xW'], '--', color=colors[exp_name],
                    linewidth=2, label=f'{exp_name} SW@10×W', alpha=0.5)
            
            # Detect overfitting at 10×W
            overfit_10x, overfit_step_10x, best_step_10x, best_ppl_10x = detect_overfitting(
                data['steps'], data['ttt_10xW'])
            
            ax2.scatter([best_step_10x], [best_ppl_10x], color=colors[exp_name], 
                       s=100, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
    
    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Perplexity (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Long Sequence Perplexity (10×W)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: SW/TTT Ratio at 10×W
    ax3 = fig.add_subplot(gs[2, :2])
    for exp_name, data in experiments.items():
        if data['ttt_10xW'][0] is not None:
            ratio = np.array(data['sw_10xW']) / np.array(data['ttt_10xW'])
            ax3.plot(data['steps'], ratio, '-', color=colors[exp_name], 
                    linewidth=2, label=f'{exp_name}', alpha=0.8)
    
    ax3.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
    ax3.fill_between(ax3.get_xlim(), 0.95, 1.05, color='gray', alpha=0.2, label='±5% band')
    ax3.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax3.set_ylabel('SW/TTT Ratio', fontsize=12, fontweight='bold')
    ax3.set_title('Length Generalization Ratio (SW/TTT at 10×W)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.6, 1.4)
    
    # Plot 4: Summary table
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.axis('off')
    
    table_data = [['Metric'] + list(experiments.keys())]
    
    for exp_name, data in experiments.items():
        overfit, overfit_step, best_step, best_ppl = detect_overfitting(
            data['steps'], data['ttt_1xW'])
        
        if exp_name not in [row[0] for row in table_data[1:]]:
            table_data.append(['Best Step'])
            table_data.append(['Best PPL@1×W'])
            table_data.append(['Final PPL@1×W'])
            table_data.append(['Overfitting?'])
            table_data.append(['Overfit Step'])
            break
    
    for i, exp_name in enumerate(experiments.keys()):
        data = experiments[exp_name]
        overfit, overfit_step, best_step, best_ppl = detect_overfitting(
            data['steps'], data['ttt_1xW'])
        
        col_idx = i + 1
        table_data[1].append(f"{best_step:,}")
        table_data[2].append(f"{best_ppl:.1f}")
        table_data[3].append(f"{data['ttt_1xW'][-1]:.1f}")
        table_data[4].append("Yes" if overfit else "No")
        table_data[5].append(f"{overfit_step:,}" if overfit else "N/A")
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('lightgray')
        table[(0, i)].set_text_props(weight='bold')
    
    ax4.set_title('Overfitting Summary\n(1×W)', fontsize=12, fontweight='bold', pad=20)
    
    # Plot 5: 10×W overfitting summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    table_10x_data = [['Metric'] + list(experiments.keys())]
    table_10x_data.extend([
        ['Best Step'],
        ['Best PPL@10×W'],
        ['Final PPL@10×W'],
        ['Overfitting?'],
    ])
    
    for i, exp_name in enumerate(experiments.keys()):
        data = experiments[exp_name]
        if data['ttt_10xW'][0] is not None:
            overfit, overfit_step, best_step, best_ppl = detect_overfitting(
                data['steps'], data['ttt_10xW'])
            
            col_idx = i + 1
            table_10x_data[1].append(f"{best_step:,}")
            table_10x_data[2].append(f"{best_ppl:.1f}")
            table_10x_data[3].append(f"{data['ttt_10xW'][-1]:.1f}")
            table_10x_data[4].append("Yes" if overfit else "No")
        else:
            for row in range(1, 5):
                table_10x_data[row].append("N/A")
    
    table_10x = ax5.table(cellText=table_10x_data, cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
    table_10x.auto_set_font_size(False)
    table_10x.set_fontsize(9)
    table_10x.scale(1, 3)
    
    for i in range(len(table_10x_data[0])):
        table_10x[(0, i)].set_facecolor('lightgray')
        table_10x[(0, i)].set_text_props(weight='bold')
    
    ax5.set_title('Overfitting Summary\n(10×W)', fontsize=12, fontweight='bold', pad=20)
    
    # Plot 6: Key insights
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    insights = ["KEY INSIGHTS:", ""]
    
    for exp_name, data in experiments.items():
        overfit_1x, overfit_step_1x, best_step_1x, best_ppl_1x = detect_overfitting(
            data['steps'], data['ttt_1xW'])
        
        insights.append(f"{exp_name}:")
        insights.append(f"  Best@1×W: {best_ppl_1x:.1f} at {best_step_1x:,}")
        
        if overfit_1x:
            degradation = (data['ttt_1xW'][-1] / best_ppl_1x - 1) * 100
            insights.append(f"  Overfits after {overfit_step_1x:,} steps")
            insights.append(f"  Degradation: +{degradation:.1f}%")
        else:
            insights.append(f"  No overfitting detected")
        
        if data['ttt_10xW'][0] is not None:
            overfit_10x, _, best_step_10x, best_ppl_10x = detect_overfitting(
                data['steps'], data['ttt_10xW'])
            
            final_ratio = data['sw_10xW'][-1] / data['ttt_10xW'][-1]
            best_ratio = data['sw_10xW'][data['steps'].index(best_step_10x)] / best_ppl_10x
            
            if final_ratio < 1.0:
                insights.append(f"  Length gen: TTT wins ({(1-final_ratio)*100:.1f}%)")
            else:
                insights.append(f"  Length gen: SW wins ({(final_ratio-1)*100:.1f}%)")
        
        insights.append("")
    
    text = '\n'.join(insights)
    ax6.text(0.1, 0.9, text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig('overfitting_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✅ Analysis plot saved: overfitting_analysis.png")
    
    # Print detailed console summary
    print("\n" + "=" * 80)
    print("DETAILED OVERFITTING ANALYSIS")
    print("=" * 80)
    
    for exp_name, data in experiments.items():
        print(f"\n{exp_name}:")
        print("-" * 40)
        
        # 1×W analysis
        overfit_1x, overfit_step_1x, best_step_1x, best_ppl_1x = detect_overfitting(
            data['steps'], data['ttt_1xW'])
        
        print(f"  Training Window (1×W):")
        print(f"    Best: {best_ppl_1x:.1f} at step {best_step_1x:,}")
        print(f"    Final: {data['ttt_1xW'][-1]:.1f} at step {data['steps'][-1]:,}")
        
        if overfit_1x:
            degradation = (data['ttt_1xW'][-1] / best_ppl_1x - 1) * 100
            print(f"    ⚠️  OVERFITTING DETECTED at step {overfit_step_1x:,}")
            print(f"    Degradation from best: +{degradation:.1f}%")
        else:
            improvement = (1 - data['ttt_1xW'][-1] / best_ppl_1x) * 100
            if improvement < 0:
                print(f"    ⚠️  Possible overfitting (no strong signal)")
                print(f"    Change from best: {improvement:.1f}%")
            else:
                print(f"    ✅ No overfitting detected")
                print(f"    Still improving: {improvement:.1f}%")
        
        # 10×W analysis
        if data['ttt_10xW'][0] is not None:
            overfit_10x, overfit_step_10x, best_step_10x, best_ppl_10x = detect_overfitting(
                data['steps'], data['ttt_10xW'])
            
            print(f"\n  Long Sequence (10×W):")
            print(f"    TTT Best: {best_ppl_10x:.1f} at step {best_step_10x:,}")
            print(f"    TTT Final: {data['ttt_10xW'][-1]:.1f}")
            print(f"    SW Final: {data['sw_10xW'][-1]:.1f}")
            
            final_ratio = data['sw_10xW'][-1] / data['ttt_10xW'][-1]
            print(f"    SW/TTT ratio: {final_ratio:.3f}", end=" ")
            
            if final_ratio < 1.0:
                print(f"(TTT wins by {(1-final_ratio)*100:.1f}%)")
            else:
                print(f"(SW wins by {(final_ratio-1)*100:.1f}%)")
            
            if overfit_10x:
                degradation_10x = (data['ttt_10xW'][-1] / best_ppl_10x - 1) * 100
                print(f"    ⚠️  OVERFITTING at 10×W after step {overfit_step_10x:,}")
                print(f"    Degradation: +{degradation_10x:.1f}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
