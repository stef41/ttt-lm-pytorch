#!/usr/bin/env python3
"""
Read a chunk perplexity JSON and print summary comparisons (TTT vs SW).
Usage:
  python compare_length_generalization.py --json path/to/chunk_eval_...json
"""
import argparse
import json
import math
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="Path to chunk_eval_..._final.json")
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.json)
    data = json.loads(path.read_text())

    W = int(data["training_window"])
    x = data.get("chunk_x") or data["chunk_centers"]
    ttt = data["chunk_ppl_ttt"]
    sw = data["chunk_ppl_sw"]

    def closest_idx(target):
        return min(range(len(x)), key=lambda i: abs(x[i] - target)) if x else None

    anchors = [1, 5, 10]
    print(f"Summary for W={W}, T={data['max_length']}, K={data['chunk_size']}, S={data['chunk_stride']}\n")
    for r in anchors:
        target = W * r
        i = closest_idx(target)
        if i is not None:
            print(f"@{r}xW (x≈{x[i]}): TTT={ttt[i]:.2f}, SW={sw[i]:.2f}, Δ={sw[i]-ttt[i]:.2f}")
    if ttt and sw:
        def summary(vals):
            return min(vals), sum(vals)/len(vals), max(vals)
        tmin, tmean, tmax = summary(ttt)
        smin, smean, smax = summary(sw)
        print("\nTTT  min/mean/max:", f"{tmin:.2f}/{tmean:.2f}/{tmax:.2f}")
        print("SW   min/mean/max:", f"{smin:.2f}/{smean:.2f}/{smax:.2f}")


if __name__ == "__main__":
    main()
