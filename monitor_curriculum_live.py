#!/usr/bin/env python3
"""
Monitor curriculum training and auto-restart if needed.
"""
import time
import subprocess
import json
from pathlib import Path
import sys

curriculum_dir = Path("curriculum_exp")
log_file = curriculum_dir / "training.log"
metadata_file = curriculum_dir / "metadata.json"

def count_checkpoints():
    """Count completed checkpoint directories."""
    if not curriculum_dir.exists():
        return 0
    return len(list(curriculum_dir.glob("checkpoint_*")))

def get_last_log_line():
    """Get last line from log file."""
    if not log_file.exists():
        return None
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            return lines[-1].strip() if lines else None
    except Exception:
        return None

def main():
    print("🔍 Monitoring curriculum training...")
    
    if not metadata_file.exists():
        print("❌ No metadata.json found. Run hasn't started yet.")
        return
    
    with open(metadata_file, "r") as f:
        meta = json.load(f)
    
    target_checkpoints = meta["num_eval_checkpoints"]
    print(f"  Target: {target_checkpoints} checkpoints")
    
    last_count = 0
    stall_time = 0
    last_log_line = None
    
    while True:
        current_count = count_checkpoints()
        current_log_line = get_last_log_line()
        
        # Progress update
        if current_count != last_count:
            print(f"✅ Progress: {current_count}/{target_checkpoints} checkpoints completed")
            last_count = current_count
            stall_time = 0
            
            if current_count >= target_checkpoints:
                print("🎉 All checkpoints complete!")
                break
        
        # Log line changed (training is active)
        if current_log_line != last_log_line:
            if current_log_line and "Step" in current_log_line:
                print(f"📊 {current_log_line}")
            last_log_line = current_log_line
            stall_time = 0
        else:
            stall_time += 30
        
        # Check for stall (no progress for 10 minutes during training)
        if current_count > 0 and current_count < target_checkpoints and stall_time > 600:
            print(f"⚠️  Training appears stalled (no progress for {stall_time}s)")
            print("   Check terminal or restart manually if needed")
        
        time.sleep(30)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏸️  Monitor stopped")
        sys.exit(0)
