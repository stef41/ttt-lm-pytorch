#!/usr/bin/env python3
"""
Hyperparameter tuner for TTT models

Workflow per trial
------------------
1) Launch training via train_gpu_optimized.py with chosen hyperparameters.
2) Wait for completion; model saved at <trial_dir>/final_model.
3) Load model + tokenizer; run chunk-perplexity evaluation vs sliding-window baseline
   at long context (e.g., 10× training window).
4) Compute a scalar score favoring TTT over sliding-window beyond the training window.
5) Track the best config; save per-trial JSON and a final summary; copy best model.

Usage
-----
python tune.py --trials 4 --search random --output-dir tuning_runs \
  --max-train-steps 50 --dataset-subset-size 5000 --eval-factor 10

Notes
-----
- Runs trials sequentially to avoid GPU contention.
- Disables W&B in training by default (override with --wandb if desired).
- Keeps model_size fixed to "125m" and max_seq_length fixed to 64 for speed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Local imports
from perplexity_evaluator import evaluate_model_perplexity
from ttt import TTTForCausalLM
from transformers import AutoTokenizer


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _now() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _device_str() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class TrialConfig:
    # Training hyperparameters (subset of train_gpu_optimized.py args)
    model_size: str = "125m"
    ttt_layer_type: str = "linear"  # ["linear", "mlp"]
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    max_seq_length: int = 64
    max_train_steps: int = 50
    mixed_precision: str = "bf16"  # "no" | "fp16" | "bf16"
    state_passing: bool = True
    state_reset_interval: int = 100
    dataset_name: str = "allenai/c4"
    dataset_config: str = "en"
    dataset_subset_size: int = -1
    seed: int = 42

    # Eval hyperparameters
    eval_factor: int = 10  # evaluate at T = eval_factor * max_seq_length
    eval_max_seqs: int = 16
    eval_batch_size: int = 4
    eval_window_batch_size: int = 128
    eval_use_packing: bool = True

    # Bookkeeping
    run_name: Optional[str] = None


def default_search_space() -> Dict[str, List[Any]]:
    return {
        "ttt_layer_type": ["linear", "mlp"],
        "learning_rate": [1e-3, 5e-4, 2e-4],
        "per_device_train_batch_size": [2, 4],
        "gradient_accumulation_steps": [2, 4],
        "state_passing": [True],  # state passing helps continuity; keep ON by default
        # keep model_size and max_seq_length fixed for speed
    }


def sample_config(base: TrialConfig, space: Dict[str, List[Any]], rng: random.Random) -> TrialConfig:
    cfg = TrialConfig(**asdict(base))
    for k, choices in space.items():
        setattr(cfg, k, rng.choice(choices))
    # Derive mixed precision based on device
    if not torch.cuda.is_available():
        cfg.mixed_precision = "no"
    else:
        # prefer bf16 on Ampere+; else fp16
        major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
        cfg.mixed_precision = "bf16" if major >= 8 else "fp16"
    return cfg


def build_train_cmd(cfg: TrialConfig, out_dir: Path, enable_wandb: bool = False) -> List[str]:
    cmd = [
        sys.executable, str(Path(__file__).parent / "train_gpu_optimized.py"),
        "--dataset_name", cfg.dataset_name,
        "--dataset_config", cfg.dataset_config,
        "--dataset_subset_size", str(cfg.dataset_subset_size),
        "--model_size", cfg.model_size,
        "--ttt_layer_type", cfg.ttt_layer_type,
        "--per_device_train_batch_size", str(cfg.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(cfg.gradient_accumulation_steps),
        "--learning_rate", str(cfg.learning_rate),
        "--max_seq_length", str(cfg.max_seq_length),
        "--max_train_steps", str(cfg.max_train_steps),
        "--mixed_precision", cfg.mixed_precision,
        "--output_dir", str(out_dir),
        "--seed", str(cfg.seed),
        "--state_reset_interval", str(cfg.state_reset_interval),
        "--eval_skip",  # we'll evaluate post-hoc in tuner
    ]
    # state passing toggles
    if cfg.state_passing:
        cmd.append("--state_passing")
    else:
        cmd.append("--no_state_passing")

    # W&B control
    if not enable_wandb:
        cmd.append("--no_wandb")
    return cmd


def train_once(cfg: TrialConfig, trial_dir: Path, enable_wandb: bool = False) -> Tuple[int, Optional[str]]:
    trial_dir.mkdir(parents=True, exist_ok=True)
    # Persist config
    with open(trial_dir / "trial_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    cmd = build_train_cmd(cfg, trial_dir, enable_wandb)
    env = os.environ.copy()
    # Make W&B fully inert when disabled
    if not enable_wandb:
        env["WANDB_MODE"] = "disabled"

    print(f"\n[trial] launching: {' '.join(cmd)}")
    start = time.time()
    # Stream stdout/stderr to a per-trial log file for debugging
    log_path = trial_dir / "train.log"
    with open(log_path, "w") as log_fp:
        proc = subprocess.run(cmd, env=env, cwd=str(Path(__file__).parent), stdout=log_fp, stderr=log_fp)
    elapsed = time.time() - start
    print(f"[trial] finished in {elapsed:.1f}s with code {proc.returncode}")
    final_model_dir = trial_dir / "final_model"
    if proc.returncode != 0:
        return proc.returncode, None
    if not final_model_dir.exists():
        print(f"[trial] ERROR: expected saved model at {final_model_dir}, but not found")
        return 1, None
    return 0, str(final_model_dir)


def evaluate_once(
    model_dir: str,
    *,
    training_window: int,
    eval_factor: int,
    eval_max_seqs: int,
    eval_batch_size: int,
    eval_window_batch_size: int,
    use_packing: bool,
    dataset_name: str,
    dataset_config: str,
    output_dir: Path,
    step: Optional[int] = None,
) -> Dict[str, Any]:
    device = _device_str()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = TTTForCausalLM.from_pretrained(model_dir)
    T = int(training_window * eval_factor)

    res = evaluate_model_perplexity(
        model=model,
        tokenizer=tokenizer,
        device=device,
        training_window=training_window,
        max_seqs=eval_max_seqs,
        batch_size=eval_batch_size,
        window_batch_size=eval_window_batch_size,
        use_amp=(device.startswith("cuda")),
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_length=T,
        output_dir=str(output_dir),
        step=step,
        log_wandb=False,
        use_packing=use_packing,
    )
    return res


def long_range_score(res: Dict[str, Any], W: int, focus_multiplier: int = 5) -> Dict[str, float]:
    """
    Compute scalar metrics emphasizing length generalization:
    - ratio_median_≥mW = median over centers ≥ m*W of (SW / TTT).
    - diff_median_≥mW  = median over centers ≥ m*W of (SW - TTT).
    - win_rate_≥mW     = fraction of centers ≥ m*W where TTT < SW.
    Larger is better for ratio and diff; win_rate close to 1 means consistent wins.
    """
    centers = res.get("chunk_centers", [])
    ttt = res.get("chunk_ppl_ttt", [])
    sw = res.get("chunk_ppl_sw", [])
    if not centers or not ttt or not sw or len(centers) != len(ttt) or len(ttt) != len(sw):
        return {"ratio_median": float("nan"), "diff_median": float("nan"), "win_rate": float("nan")}

    m = int(focus_multiplier)
    # chunk centers are in token-loss index space, and each chunk spans K tokens.
    # By default K == W in evaluator; use center threshold adjusted by half chunk length.
    # We don't have K here, but default is W, so require c >= m*W - W//2 to consider chunks fully at or beyond m*W.
    cutoff_center = m * W - (W // 2)
    idxs = [i for i, c in enumerate(centers) if c >= cutoff_center]
    if not idxs:
        # Fallback: use last quarter of centers
        n = max(1, len(centers) // 4)
        idxs = list(range(max(0, len(centers) - n), len(centers)))

    ratios = [sw[i] / max(1e-8, ttt[i]) for i in idxs]
    diffs = [sw[i] - ttt[i] for i in idxs]
    wins = [1.0 if ttt[i] < sw[i] else 0.0 for i in idxs]

    ratios_sorted = sorted(ratios)
    diffs_sorted = sorted(diffs)
    mid = len(idxs) // 2
    ratio_med = ratios_sorted[mid] if ratios_sorted else float("nan")
    diff_med = diffs_sorted[mid] if diffs_sorted else float("nan")
    win_rate = sum(wins) / len(wins) if wins else float("nan")
    return {"ratio_median": ratio_med, "diff_median": diff_med, "win_rate": win_rate}


def choose_score(metrics: Dict[str, float]) -> float:
    """Single scalar to maximize. Prefer ratio; fallback to diff; else NaN."""
    r = _safe_float(metrics.get("ratio_median"))
    if math.isfinite(r):
        return r
    d = _safe_float(metrics.get("diff_median"))
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=4, help="Number of trials to run")
    parser.add_argument("--search", type=str, default="random", choices=["random", "grid"])
    parser.add_argument("--output-dir", type=str, default="tuning_runs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-train-steps", type=int, default=50)
    parser.add_argument("--dataset-subset-size", type=int, default=5000)
    parser.add_argument("--dataset-name", type=str, default="allenai/c4")
    parser.add_argument("--dataset-config", type=str, default="en")
    parser.add_argument("--eval-factor", type=int, default=10)
    parser.add_argument("--eval-max-seqs", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-window-batch-size", type=int, default=128)
    parser.add_argument("--focus-multiplier", type=int, default=5, help="Compute score using centers ≥ m×W")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B in training runs")
    parser.add_argument("--dry-run", action="store_true", help="Skip training and only print planned trials")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    base = TrialConfig()
    base.max_train_steps = args.max_train_steps
    base.dataset_subset_size = args.dataset_subset_size
    base.dataset_name = args.dataset_name
    base.dataset_config = args.dataset_config

    space = default_search_space()

    # Precompute grid if requested
    grid: List[TrialConfig] = []
    if args.search == "grid":
        for ttt_layer_type in space["ttt_layer_type"]:
            for lr in space["learning_rate"]:
                for bsz in space["per_device_train_batch_size"]:
                    for gas in space["gradient_accumulation_steps"]:
                        for sp in space["state_passing"]:
                            cfg = TrialConfig(**asdict(base))
                            cfg.ttt_layer_type = ttt_layer_type
                            cfg.learning_rate = lr
                            cfg.per_device_train_batch_size = bsz
                            cfg.gradient_accumulation_steps = gas
                            cfg.state_passing = sp
                            # set precision based on device
                            if not torch.cuda.is_available():
                                cfg.mixed_precision = "no"
                            else:
                                major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
                                cfg.mixed_precision = "bf16" if major >= 8 else "fp16"
                            grid.append(cfg)

    summary: Dict[str, Any] = {
        "started_at": _now(),
        "device": _device_str(),
        "trials_planned": args.trials if args.search == "random" else min(args.trials, len(grid)),
        "search": args.search,
        "results": [],
        "best": None,
    }

    best_score = -float("inf")
    best_trial_dir: Optional[Path] = None

    for i in range(summary["trials_planned"]):
        cfg = sample_config(base, space, rng) if args.search == "random" else grid[i]
        cfg.run_name = cfg.run_name or f"trial_{i:02d}_{_now()}"

        trial_dir = root / cfg.run_name
        print(f"\n=== Trial {i+1}/{summary['trials_planned']} → {trial_dir.name} ===")
        print(json.dumps(asdict(cfg), indent=2))

        if args.dry_run:
            # Only write planned config and continue
            trial_dir.mkdir(parents=True, exist_ok=True)
            with open(trial_dir / "trial_config.json", "w") as f:
                json.dump(asdict(cfg), f, indent=2)
            continue

        rc, model_dir = train_once(cfg, trial_dir, enable_wandb=args.wandb)
        if rc != 0 or not model_dir:
            result = {
                "status": "failed",
                "returncode": rc,
            }
            with open(trial_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2)
            summary["results"].append({"trial": cfg.run_name, **result})
            continue

        # Evaluate
        try:
            res = evaluate_once(
                model_dir,
                training_window=cfg.max_seq_length,
                eval_factor=args.eval_factor,
                eval_max_seqs=args.eval_max_seqs,
                eval_batch_size=args.eval_batch_size,
                eval_window_batch_size=args.eval_window_batch_size,
                use_packing=cfg.eval_use_packing,
                dataset_name=cfg.dataset_name,
                dataset_config=cfg.dataset_config,
                output_dir=trial_dir,
                step=cfg.max_train_steps,
            )
            metrics = long_range_score(res, W=cfg.max_seq_length, focus_multiplier=args.focus_multiplier)
            score = choose_score(metrics)
            result = {
                "status": "ok",
                "model_dir": model_dir,
                "eval": res,
                "metrics": metrics,
                "score": score,
            }
        except Exception as e:
            # Persist evaluation error to file for easier inspection
            with open(trial_dir / "eval_error.txt", "w") as ef:
                ef.write(str(e))
            result = {"status": "eval_failed", "error": str(e)}

        with open(trial_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        summary_entry = {"trial": cfg.run_name, "config": asdict(cfg), **result}
        summary["results"].append(summary_entry)

        # Track best
        if result.get("status") == "ok":
            sc = _safe_float(result.get("score"), -float("inf"))
            if sc > best_score:
                best_score = sc
                best_trial_dir = trial_dir
                summary["best"] = {
                    "trial": cfg.run_name,
                    "score": best_score,
                    "metrics": result.get("metrics"),
                    "model_dir": model_dir,
                }

        # Persist intermediate summary
        with open(root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Finalize best model copy
    if best_trial_dir and (best_trial_dir / "final_model").exists():
        dst = root / "best_model"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(best_trial_dir / "final_model", dst)
        summary["best_model_path"] = str(dst)

    # Save final summary
    summary["finished_at"] = _now()
    with open(root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTuning complete.")
    if summary.get("best"):
        print(f"Best trial: {summary['best']['trial']} score={summary['best']['score']:.3f}")
        print(f"Best model: {summary.get('best_model_path', 'N/A')}")
    else:
        print("No successful trials.")


if __name__ == "__main__":
    main()
