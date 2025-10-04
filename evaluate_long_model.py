#!/usr/bin/env python3
"""
Evaluate a saved TTT model for length generalization using chunk perplexity.

Defaults:
- W (training window) = 64
- T (max_length) = 1280 (20xW)
- K (chunk size) = 16
- S (chunk stride) = 16
- x-axis = 'end' (plot points at chunk ends: 16, 32, 48, ...)

Saves a plot and a JSON with the raw results into --output_dir.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ttt import TTTForCausalLM
from perplexity_evaluator import evaluate_model_perplexity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True, help="Path to saved model directory (with config.json)")
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save plots/results; default: parent of model_dir")
    p.add_argument("--W", type=int, default=64, help="Training window size used for the model")
    p.add_argument("--T", type=int, default=1280, help="Evaluation max length (e.g., 20xW)")
    p.add_argument("--K", type=int, default=16, help="Chunk size")
    p.add_argument("--S", type=int, default=16, help="Chunk stride")
    p.add_argument("--max_seqs", type=int, default=32, help="Max sequences to evaluate")
    p.add_argument("--batch_size", type=int, default=8, help="Full-context eval batch size")
    p.add_argument("--window_batch_size", type=int, default=128, help="Sliding-window eval batch size")
    p.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name for evaluation")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1", help="Dataset config for evaluation")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu; default: auto")
    p.add_argument("--no_packing", action="store_true", help="Disable packing and use naive doc truncation")
    return p.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in model_dir: {model_dir}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else model_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer and model
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = TTTForCausalLM.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    # Run evaluation
    res = evaluate_model_perplexity(
        model=model,
        tokenizer=tok,
        device=device,
        training_window=int(args.W),
        max_seqs=int(args.max_seqs),
        batch_size=int(args.batch_size),
        window_batch_size=int(args.window_batch_size),
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        max_length=int(args.T),
        chunk_size=int(args.K),
        chunk_stride=int(args.S),
        output_dir=str(out_dir),
        step="final",
        log_wandb=False,
        use_packing=(not args.no_packing),
        align_prefix_pre_w=True,
        x_axis="end",
    )

    # Save raw results JSON
    json_path = out_dir / f"chunk_eval_W{res['training_window']}_T{res['max_length']}_K{res['chunk_size']}_S{res['chunk_stride']}_final.json"
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved results JSON: {json_path}")


if __name__ == "__main__":
    main()
