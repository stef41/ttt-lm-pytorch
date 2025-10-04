#!/usr/bin/env python3
"""
Quick helper to monitor curriculum training progress.

Usage:
  python monitor_curriculum.py --dir curriculum_exp
"""
import argparse
import json
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="curriculum_exp", help="Curriculum output dir")
    return p.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.dir)
    
    if not exp_dir.exists():
        print(f"Directory not found: {exp_dir}")
        return
    
    # Check metadata
    meta_path = exp_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print("📋 Experiment Configuration:")
        print(f"  Training window: {meta.get('training_window')}")
        print(f"  Eval max length: {meta.get('eval_max_length')}")
        print(f"  Max train steps: {meta.get('max_train_steps')}")
        print(f"  Num eval checkpoints: {meta.get('num_eval_checkpoints')}")
        print(f"  Learning rate: {meta.get('learning_rate')}")
        print()
    
    # Check all_summaries.json
    summaries_path = exp_dir / "all_summaries.json"
    if summaries_path.exists():
        with open(summaries_path, "r") as f:
            summaries = json.load(f)
        print(f"📊 Completed Checkpoints: {len(summaries)}")
        
        if summaries:
            print("\nRecent checkpoints:")
            for s in summaries[-3:]:
                step = s.get("step", "?")
                ttt_10x = s.get("ttt_ppl_at_10xW", "?")
                sw_10x = s.get("sw_ppl_at_10xW", "?")
                print(f"  Step {step}: TTT@10xW={ttt_10x}, SW@10xW={sw_10x}")
    else:
        print("⏳ Waiting for first checkpoint...")
    
    # Count checkpoint directories
    checkpoint_dirs = sorted(exp_dir.glob("checkpoint_*"))
    print(f"\n📁 Checkpoint directories: {len(checkpoint_dirs)}")
    
    # Check if final model exists
    final_model_dir = exp_dir / "final_model"
    if final_model_dir.exists():
        print("✅ Final model saved")
    else:
        print("⏳ Training in progress...")


if __name__ == "__main__":
    main()
