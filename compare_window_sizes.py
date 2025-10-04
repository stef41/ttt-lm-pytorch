#!/usr/bin/env python3
"""
Compare curriculum experiments with different training windows.

Shows side-by-side comparison of W=64 vs W=128 experiments.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_experiment_data(exp_dir):
    """Load experiment metadata and summaries."""
    exp_dir = Path(exp_dir)
    
    if not exp_dir.exists():
        return None
    
    # Load metadata
    meta_path = exp_dir / "metadata.json"
    if not meta_path.exists():
        return None
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Load summaries
    summaries_path = exp_dir / "all_summaries.json"
    summaries = []
    if summaries_path.exists():
        with open(summaries_path) as f:
            summaries = json.load(f)
    
    return {
        'meta': meta,
        'summaries': summaries,
        'n_checkpoints': len(summaries),
    }


def format_ppl(ppl):
    """Format perplexity value."""
    if ppl < 1000:
        return f"{ppl:.1f}"
    elif ppl < 10000:
        return f"{ppl/1000:.1f}k"
    else:
        return f"{ppl/1000:.0f}k"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--w64_dir", type=str, default="curriculum_exp")
    p.add_argument("--w128_dir", type=str, default="curriculum_exp_w128")
    args = p.parse_args()
    
    print("\n" + "="*80)
    print("Curriculum Training Comparison: W=64 vs W=128")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load experiments
    w64 = load_experiment_data(args.w64_dir)
    w128 = load_experiment_data(args.w128_dir)
    
    if not w64 and not w128:
        print("❌ No experiments found!")
        return
    
    # Configuration comparison
    print("📊 Configuration Comparison\n")
    print(f"{'Metric':<30} {'W=64':<20} {'W=128':<20}")
    print("-" * 70)
    
    if w64:
        w64_window = w64['meta']['training_window']
        w64_eval_len = w64['meta']['eval_max_length']
        w64_steps = w64['meta']['max_train_steps']
        w64_ckpts = w64['meta']['num_eval_checkpoints']
    else:
        w64_window = w64_eval_len = w64_steps = w64_ckpts = "N/A"
    
    if w128:
        w128_window = w128['meta']['training_window']
        w128_eval_len = w128['meta']['eval_max_length']
        w128_steps = w128['meta']['max_train_steps']
        w128_ckpts = w128['meta']['num_eval_checkpoints']
    else:
        w128_window = w128_eval_len = w128_steps = w128_ckpts = "N/A"
    
    print(f"{'Training Window (W)':<30} {w64_window:<20} {w128_window:<20}")
    print(f"{'Eval Max Length (T)':<30} {w64_eval_len:<20} {w128_eval_len:<20}")
    print(f"{'Max Train Steps':<30} {w64_steps:<20} {w128_steps:<20}")
    print(f"{'Eval Checkpoints':<30} {w64_ckpts:<20} {w128_ckpts:<20}")
    
    # Progress comparison
    print(f"\n📈 Progress Comparison\n")
    print(f"{'Metric':<30} {'W=64':<20} {'W=128':<20}")
    print("-" * 70)
    
    if w64 and w64['summaries']:
        w64_latest = w64['summaries'][-1]
        w64_step = w64_latest['step']
        w64_n_ckpts = len(w64['summaries'])
        w64_pct = 100 * w64_step / w64['meta']['max_train_steps']
        w64_progress = f"{w64_step:,} ({w64_pct:.1f}%)"
        w64_ckpt_progress = f"{w64_n_ckpts}/{w64['meta']['num_eval_checkpoints']}"
    else:
        w64_progress = "Not started"
        w64_ckpt_progress = "0/?"
    
    if w128 and w128['summaries']:
        w128_latest = w128['summaries'][-1]
        w128_step = w128_latest['step']
        w128_n_ckpts = len(w128['summaries'])
        w128_pct = 100 * w128_step / w128['meta']['max_train_steps']
        w128_progress = f"{w128_step:,} ({w128_pct:.1f}%)"
        w128_ckpt_progress = f"{w128_n_ckpts}/{w128['meta']['num_eval_checkpoints']}"
    else:
        w128_progress = "Not started"
        w128_ckpt_progress = "0/?"
    
    print(f"{'Current Step':<30} {w64_progress:<20} {w128_progress:<20}")
    print(f"{'Checkpoints':<30} {w64_ckpt_progress:<20} {w128_ckpt_progress:<20}")
    
    # Performance comparison (if both have data)
    if w64 and w64['summaries'] and w128 and w128['summaries']:
        print(f"\n🏆 Performance Comparison (Latest Checkpoint)\n")
        print(f"{'Metric':<30} {'W=64':<20} {'W=128':<20} {'Change':<15}")
        print("-" * 85)
        
        w64_latest = w64['summaries'][-1]
        w128_latest = w128['summaries'][-1]
        
        # @ 1×W
        if 'ttt_ppl_at_1xW' in w64_latest and 'ttt_ppl_at_1xW' in w128_latest:
            w64_1x = w64_latest['ttt_ppl_at_1xW']
            w128_1x = w128_latest['ttt_ppl_at_1xW']
            change_1x = ((w128_1x - w64_1x) / w64_1x) * 100
            change_str = f"{change_1x:+.1f}%"
            print(f"{'TTT PPL @ 1×W':<30} {format_ppl(w64_1x):<20} {format_ppl(w128_1x):<20} {change_str:<15}")
        
        # @ 10×W
        if 'ttt_ppl_at_10xW' in w64_latest and 'ttt_ppl_at_10xW' in w128_latest:
            w64_10x_ttt = w64_latest['ttt_ppl_at_10xW']
            w128_10x_ttt = w128_latest['ttt_ppl_at_10xW']
            w64_10x_sw = w64_latest['sw_ppl_at_10xW']
            w128_10x_sw = w128_latest['sw_ppl_at_10xW']
            
            change_10x_ttt = ((w128_10x_ttt - w64_10x_ttt) / w64_10x_ttt) * 100
            change_10x_sw = ((w128_10x_sw - w64_10x_sw) / w64_10x_sw) * 100
            
            print(f"{'TTT PPL @ 10×W':<30} {format_ppl(w64_10x_ttt):<20} {format_ppl(w128_10x_ttt):<20} {change_10x_ttt:+.1f}%")
            print(f"{'SW PPL @ 10×W':<30} {format_ppl(w64_10x_sw):<20} {format_ppl(w128_10x_sw):<20} {change_10x_sw:+.1f}%")
            
            w64_ratio = w64_10x_sw / w64_10x_ttt
            w128_ratio = w128_10x_sw / w128_10x_ttt
            print(f"{'SW/TTT Ratio @ 10×W':<30} {w64_ratio:<20.3f} {w128_ratio:<20.3f}")
            
            w64_winner = "SW wins" if w64_ratio < 1.0 else "TTT wins"
            w128_winner = "SW wins" if w128_ratio < 1.0 else "TTT wins"
            print(f"{'Winner @ 10×W':<30} {w64_winner:<20} {w128_winner:<20}")
        
        # @ 20×W (if available for both)
        if 'ttt_ppl_at_20xW' in w64_latest and 'ttt_ppl_at_20xW' in w128_latest:
            w64_20x_ttt = w64_latest['ttt_ppl_at_20xW']
            w128_20x_ttt = w128_latest['ttt_ppl_at_20xW']
            w64_20x_sw = w64_latest['sw_ppl_at_20xW']
            w128_20x_sw = w128_latest['sw_ppl_at_20xW']
            
            change_20x_ttt = ((w128_20x_ttt - w64_20x_ttt) / w64_20x_ttt) * 100
            
            print(f"{'TTT PPL @ 20×W':<30} {format_ppl(w64_20x_ttt):<20} {format_ppl(w128_20x_ttt):<20} {change_20x_ttt:+.1f}%")
            print(f"{'SW PPL @ 20×W':<30} {format_ppl(w64_20x_sw):<20} {format_ppl(w128_20x_sw):<20}")
    
    # Status summary
    print(f"\n📝 Status Summary\n")
    
    if w64:
        w64_status = "✅ Complete" if w64.get('meta', {}).get('early_stopped') or len(w64['summaries']) >= w64['meta']['num_eval_checkpoints'] else "🟢 Running"
        print(f"W=64:  {w64_status}")
        if w64['summaries']:
            best_idx = w64['meta'].get('best_checkpoint_idx', -1)
            if best_idx >= 0 and best_idx < len(w64['summaries']):
                best_ppl = w64['meta'].get('best_ppl_at_1xW', 0)
                print(f"       Best checkpoint: {best_idx} (PPL@1×W = {format_ppl(best_ppl)})")
    
    if w128:
        w128_status = "✅ Complete" if w128.get('meta', {}).get('early_stopped') or len(w128['summaries']) >= w128['meta']['num_eval_checkpoints'] else "🟢 Running"
        print(f"W=128: {w128_status}")
        if w128['summaries']:
            best_idx = w128['meta'].get('best_checkpoint_idx', -1)
            if best_idx >= 0 and best_idx < len(w128['summaries']):
                best_ppl = w128['meta'].get('best_ppl_at_1xW', 0)
                print(f"       Best checkpoint: {best_idx} (PPL@1×W = {format_ppl(best_ppl)})")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
