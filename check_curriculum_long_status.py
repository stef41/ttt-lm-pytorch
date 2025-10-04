#!/usr/bin/env python3
"""
Monitor the extended curriculum training run.

Shows:
- Current training progress
- Checkpoints completed
- Best model so far
- Early stopping status
- Estimated time remaining
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--curriculum_dir", type=str, default="curriculum_exp_long")
    p.add_argument("--loop", action="store_true", help="Loop continuously")
    p.add_argument("--interval", type=int, default=180, help="Loop interval (seconds)")
    return p.parse_args()


def format_time(seconds):
    """Format seconds to human readable."""
    if seconds < 0:
        return "Unknown"
    return str(timedelta(seconds=int(seconds)))


def main():
    args = parse_args()
    curriculum_dir = Path(args.curriculum_dir)
    
    while True:
        print("\n" + "="*80)
        print(f"Extended Curriculum Training Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        if not curriculum_dir.exists():
            print(f"❌ Directory not found: {curriculum_dir}")
            if not args.loop:
                break
            time.sleep(args.interval)
            continue
        
        # Read metadata
        meta_path = curriculum_dir / "metadata.json"
        if not meta_path.exists():
            print("⏳ Training not started yet (no metadata.json)")
            if not args.loop:
                break
            time.sleep(args.interval)
            continue
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        # Count checkpoints
        checkpoint_dirs = list(curriculum_dir.glob("checkpoint_*"))
        n_checkpoints = len(checkpoint_dirs)
        
        # Read summaries
        summaries_path = curriculum_dir / "all_summaries.json"
        summaries = []
        if summaries_path.exists():
            with open(summaries_path) as f:
                summaries = json.load(f)
        
        # Basic info
        print(f"\n📊 Configuration:")
        print(f"   Training window (W): {meta['training_window']}")
        print(f"   Eval max length (T): {meta['eval_max_length']}")
        print(f"   Max train steps: {meta['max_train_steps']:,}")
        print(f"   Target checkpoints: {meta['num_eval_checkpoints']}")
        print(f"   Early stop patience: {meta['early_stop_patience']}")
        print(f"   Learning rate: {meta['learning_rate']}")
        
        # Progress
        print(f"\n📈 Progress:")
        if summaries:
            latest = summaries[-1]
            step = latest['step']
            pct = 100 * step / meta['max_train_steps']
            print(f"   Current step: {step:,} / {meta['max_train_steps']:,} ({pct:.1f}%)")
            print(f"   Checkpoints evaluated: {len(summaries)} / {meta['num_eval_checkpoints']}")
            
            # Best model
            best_idx = meta.get('best_checkpoint_idx', -1)
            best_ppl = meta.get('best_ppl_at_1xW', float('inf'))
            if best_idx >= 0 and best_idx < len(summaries):
                best_ckpt = summaries[best_idx]
                print(f"\n🏆 Best Model:")
                print(f"   Checkpoint: {best_idx} (Step {best_ckpt['step']:,})")
                print(f"   TTT PPL @ 1×W: {best_ppl:.1f}")
                if 'ttt_ppl_at_10xW' in best_ckpt:
                    print(f"   TTT PPL @ 10×W: {best_ckpt['ttt_ppl_at_10xW']:.1f}")
                    print(f"   SW PPL @ 10×W: {best_ckpt['sw_ppl_at_10xW']:.1f}")
                    ratio = best_ckpt['sw_ppl_at_10xW'] / best_ckpt['ttt_ppl_at_10xW']
                    print(f"   SW/TTT ratio: {ratio:.3f}")
            
            # Latest checkpoint
            print(f"\n📊 Latest Checkpoint ({len(summaries)-1}):")
            print(f"   Step: {latest['step']:,}")
            if 'ttt_ppl_at_1xW' in latest:
                print(f"   TTT PPL @ 1×W: {latest['ttt_ppl_at_1xW']:.1f}")
            if 'ttt_ppl_at_10xW' in latest:
                print(f"   TTT PPL @ 10×W: {latest['ttt_ppl_at_10xW']:.1f}")
                print(f"   SW PPL @ 10×W: {latest['sw_ppl_at_10xW']:.1f}")
                ratio = latest['sw_ppl_at_10xW'] / latest['ttt_ppl_at_10xW']
                winner = "SW wins" if ratio < 1.0 else "TTT wins"
                print(f"   SW/TTT ratio: {ratio:.3f} ({winner})")
            
            # Progress trend (last 5 checkpoints)
            if len(summaries) >= 5:
                recent = summaries[-5:]
                ppls = [s.get('ttt_ppl_at_1xW', 0) for s in recent if 'ttt_ppl_at_1xW' in s]
                if len(ppls) >= 2:
                    trend = ppls[-1] - ppls[0]
                    trend_str = f"↓ {abs(trend):.1f}" if trend < 0 else f"↑ {abs(trend):.1f}"
                    print(f"\n📉 Trend (last 5 ckpts): {trend_str} PPL @ 1×W")
            
            # Time estimation
            if 'start_time' in meta and step > 0:
                start = datetime.fromisoformat(meta['start_time'])
                elapsed = (datetime.now() - start).total_seconds()
                steps_per_sec = step / elapsed
                remaining_steps = meta['max_train_steps'] - step
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else -1
                
                print(f"\n⏱️  Timing:")
                print(f"   Elapsed: {format_time(elapsed)}")
                print(f"   Speed: {steps_per_sec:.2f} steps/sec")
                print(f"   ETA: {format_time(eta_seconds)}")
                
                if eta_seconds > 0:
                    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                    print(f"   Expected completion: {eta_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("   No checkpoints evaluated yet")
        
        # Check log for recent activity
        log_path = curriculum_dir / "training.log"
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    # Find last few step logs
                    recent_steps = [l for l in lines[-100:] if 'Step ' in l and 'loss=' in l]
                    if recent_steps:
                        print(f"\n📝 Recent Activity:")
                        for line in recent_steps[-3:]:
                            line = line.strip()
                            if line:
                                # Extract key info
                                if 'Step' in line and 'loss=' in line:
                                    print(f"   {line}")
            except:
                pass
        
        # Early stopping status
        if meta.get('early_stopped', False):
            print(f"\n🛑 EARLY STOPPED!")
            print(f"   Training converged at checkpoint {best_idx}")
        elif summaries and len(summaries) >= meta['num_eval_checkpoints']:
            print(f"\n✅ TRAINING COMPLETE!")
            print(f"   All {meta['num_eval_checkpoints']} checkpoints evaluated")
        
        print("\n" + "="*80)
        
        if not args.loop:
            break
        
        print(f"\n⏳ Next update in {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
