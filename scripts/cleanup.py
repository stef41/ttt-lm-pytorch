#!/usr/bin/env python3
"""
Safe cleanup utility: move generated artifacts and bulky experiment folders
to a timestamped .trash/ directory at the repo root so the workspace is clean
but recoverable.

What it moves (if present):
- Caches and trackers: __pycache__, wandb/, wandb_*/
- Output and test dirs: outputs/, test_output/, test_perplexity_output/,
  test_sliding_window_output/, comparison_outputs/, gpu_test_outputs/
- Demo/throwaway model dirs: no_bias_state_test/, no_bias_test/,
  no_state_passing_model/, state_passing_model/, properly_trained_model/,
  stable_trained_model/, well_trained_model/, conservative_ttt_model/
- Logs: *.log at repo root

It does NOT delete source files or documentation. Everything is moved, not removed.
"""

import os
import glob
import shutil
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


DIR_GLOBS = [
    "__pycache__",
    "wandb",
    "wandb_*",
    "outputs",
    "test_output",
    "test_perplexity_output",
    "test_sliding_window_output",
    "comparison_outputs",
    "gpu_test_outputs",
    "wandb_gpu_test",
    "wandb_test",
    "wandb_test_enabled",
    # model/demo artifacts likely safe to remove
    "no_bias_state_test",
    "no_bias_test",
    "no_state_passing_model",
    "state_passing_model",
    "properly_trained_model",
    "stable_trained_model",
    "well_trained_model",
    "conservative_ttt_model",
    # tuning/eval runs
    "tuning_runs",
    "tuning_runs_*",
]

FILE_GLOBS = [
    "*.log",
]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def move_path(src: str, trash_root: str):
    base = os.path.basename(src.rstrip(os.sep))
    dst = os.path.join(trash_root, base)
    # avoid collisions
    i = 2
    orig_dst = dst
    while os.path.exists(dst):
        dst = f"{orig_dst}-{i}"
        i += 1
    print(f"→ moving {src} -> {dst}")
    shutil.move(src, dst)


def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    trash_root = os.path.join(ROOT, ".trash", ts)
    ensure_dir(trash_root)

    # Directories
    moved = []
    for pat in DIR_GLOBS:
        for match in glob.glob(os.path.join(ROOT, pat)):
            if not os.path.exists(match):
                continue
            # safety: only move directories
            if os.path.isdir(match):
                move_path(match, trash_root)
                moved.append(match)

    # Files at repo root
    for pat in FILE_GLOBS:
        for match in glob.glob(os.path.join(ROOT, pat)):
            if not os.path.exists(match):
                continue
            if os.path.isfile(match):
                move_path(match, trash_root)
                moved.append(match)

    print("\nCleanup complete.")
    print(f"Moved {len(moved)} paths to {trash_root}")
    if moved:
        for m in moved:
            print(" -", os.path.relpath(m, ROOT))
        print("\nTo restore, move paths back from:")
        print(" ", trash_root)


if __name__ == "__main__":
    main()
