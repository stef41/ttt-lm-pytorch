#!/usr/bin/env python3
"""
Test script for the CHUNK-ONLY evaluator:
- Compares ChunkPPL_TTT (full-context) vs ChunkPPL_SW (sliding-window baseline)
- Prints a table and checks the generated plot in the output dir
"""

import os
import glob
import math
import logging
from typing import Optional

import torch
from transformers import AutoTokenizer
from ttt import TTTForCausalLM

from perplexity_evaluator import evaluate_model_perplexity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _fmt_range(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return "n/a"
    return f"{min(vals):.1f} - {max(vals):.1f}"


def test_sliding_window_comparison() -> bool:
    """Test the chunk-only evaluation (TTT vs SW)."""
    print("🧪 Testing ChunkPPL (TTT vs Sliding-Window)")

    # --- config ---
    model_path = "./outputs/full_dataset_training/final_model"
    W = 64
    output_dir = "./test_sliding_window_output"
    step_tag = "sliding_window_test"

    # --- load model / tokenizer ---
    print("📥 Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = TTTForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"✅ Model loaded on device: {device}")

        # --- run evaluation (chunk-only API) ---
        print("🔍 Running chunk-only evaluation (TTT vs SW baseline)...")
        results = evaluate_model_perplexity(
            model=model,
            tokenizer=tokenizer,
            device=device,
            training_window=W,           # window size for SW baseline and default chunk_size
            max_seqs=64,                 # sequences per eval
            batch_size=16,               # TTT forward batch size
            window_batch_size=256,       # SW window batch size
            use_amp=True,
            dataset_name="wikitext",
            dataset_config="wikitext-2-v1",
            max_length=None,             # None => evaluate at 10x W
            chunk_size=None,             # None => W
            chunk_stride=None,           # None => W//4
            output_dir=output_dir,
            step=step_tag,
            log_wandb=False,
        )

        # Required keys in the new API
        required = ["chunk_centers", "chunk_ppl_ttt", "chunk_ppl_sw",
                    "training_window", "max_length", "chunk_size", "chunk_stride"]
        missing = [k for k in required if k not in results]
        if missing:
            raise RuntimeError(f"Missing results keys: {missing}")

        centers = results["chunk_centers"]
        ttt = results["chunk_ppl_ttt"]
        sw  = results["chunk_ppl_sw"]
        W   = results["training_window"]
        T   = results["max_length"]
        K   = results["chunk_size"]
        S   = results["chunk_stride"]

        if not centers or not ttt or not sw:
            raise RuntimeError("Empty chunk results — need chunk_centers, chunk_ppl_ttt, chunk_ppl_sw.")

        print("\n📊 Chunk Evaluation Results:")
        print(f"✅ Chunks: {len(centers)} (K={K}, stride={S}), max length T={T}, window W={W}")

        print("\n📈 ChunkPPL (TTT vs Sliding-Window):")
        print("Center |   TTT PPL |     SW PPL | Improvement")
        print("-------|-----------|-----------|------------")
        improvements = []
        for c, t_val, s_val in zip(centers, ttt, sw):
            if math.isfinite(t_val) and math.isfinite(s_val) and s_val > 0:
                imp = (s_val - t_val) / s_val * 100.0
                improvements.append(imp)
                print(f"{c:6d} | {t_val:9.1f} | {s_val:9.1f} | {imp:+7.1f}% ({c/max(1,W):.1f}xW)")
            else:
                print(f"{c:6d} | {t_val:9.1f} | {s_val:9.1f} |     n/a   ({c/max(1,W):.1f}xW)")

        avg_improvement = (sum(improvements) / len(improvements)) if improvements else float("nan")
        print(f"\n🎯 Summary:")
        print(f"   Average TTT improvement: {avg_improvement:+.1f}%")
        print(f"   TTT chunk PPL range:     {_fmt_range(ttt)}")
        print(f"   SW  chunk PPL range:     {_fmt_range(sw)}")
        print(f"   Sequences evaluated:     {results.get('num_sequences', 'n/a')}")

        # --- check for plot file produced by new API ---
        # New plot filename pattern: chunk_eval_W{W}_T{T}_K{K}_S{S}_{step}.png
        pattern = os.path.join(output_dir, f"chunk_eval_W{W}_T{T}_K{K}_S{S}_{step_tag}*.png")
        matches = glob.glob(pattern)
        if matches:
            print(f"📊 Plot generated successfully: {matches[0]}")
        else:
            print(f"⚠️  Plot not found (pattern: {pattern})")
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                print(f"   Files in {output_dir}: {files}")

        print("🎉 Sliding window chunk comparison test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_sliding_window_comparison()
