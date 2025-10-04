#!/usr/bin/env python3
"""
Comprehensive status checker for curriculum training.
"""
import json
import time
from pathlib import Path
from datetime import datetime

def check_status():
    curriculum_dir = Path("curriculum_exp")
    
    if not curriculum_dir.exists():
        print("❌ curriculum_exp directory not found")
        return False
    
    # Load metadata
    meta_path = curriculum_dir / "metadata.json"
    if not meta_path.exists():
        print("❌ metadata.json not found")
        return False
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    target = meta["num_eval_checkpoints"]
    eval_steps = meta["eval_steps"]
    
    # Count completed checkpoints
    checkpoints = sorted(curriculum_dir.glob("checkpoint_*"))
    completed = len(checkpoints)
    
    print(f"\n{'='*60}")
    print(f"📊 Curriculum Training Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Progress: {completed}/{target} checkpoints ({100*completed/target:.1f}%)")
    
    if completed > 0:
        print(f"Latest checkpoint: {checkpoints[-1].name}")
        
        # Check latest summary
        latest_summary = checkpoints[-1] / "summary.json"
        if latest_summary.exists():
            with open(latest_summary, "r") as f:
                summary = json.load(f)
            print(f"  Step: {summary.get('step', 'N/A')}")
            print(f"  TTT@1xW: {summary.get('ttt_ppl_at_1xW', 'N/A'):.1f}" if summary.get('ttt_ppl_at_1xW') else "")
            print(f"  TTT@10xW: {summary.get('ttt_ppl_at_10xW', 'N/A'):.1f}" if summary.get('ttt_ppl_at_10xW') else "")
    
    # Check log file for recent activity
    log_file = curriculum_dir / "training.log"
    if log_file.exists():
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                if lines:
                    last_log = lines[-1].strip()
                    if "Step" in last_log:
                        print(f"\nRecent: {last_log[-100:]}")
        except Exception as e:
            print(f"⚠️  Error reading log: {e}")
    
    # Estimate completion
    if completed > 0 and completed < target:
        next_checkpoint = eval_steps[completed] if completed < len(eval_steps) else eval_steps[-1]
        print(f"\nNext checkpoint at step: {next_checkpoint}")
        remaining = target - completed
        print(f"Remaining: {remaining} checkpoints")
    elif completed >= target:
        print("\n✅ ALL CHECKPOINTS COMPLETE!")
        return True
    
    print(f"{'='*60}\n")
    return completed >= target

if __name__ == "__main__":
    import sys
    
    # Run once or in loop
    if "--loop" in sys.argv:
        print("🔄 Running in loop mode (Ctrl+C to stop)...")
        try:
            while True:
                done = check_status()
                if done:
                    print("🎉 Training complete! Exiting.")
                    sys.exit(0)
                time.sleep(180)  # Check every 3 minutes
        except KeyboardInterrupt:
            print("\n⏹️  Stopped")
    else:
        check_status()
