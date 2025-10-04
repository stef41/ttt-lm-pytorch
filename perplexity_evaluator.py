#!/usr/bin/env python3
"""
Chunk-only perplexity evaluator for TTT models (RNN-friendly).

What it measures
----------------
- ChunkPPL_TTT (full-context): One max-length forward → per-token NLL →
  slide fixed-size chunks (size=K, stride=S), mean over tokens, median over sequences, exp → PPL.
- ChunkPPL_SW (sliding-window baseline): One window pass limiting context to W
  for every position → per-token NLL with context≤W → same chunking → baseline ChunkPPL.

Key features
------------
- Single-pass (TTT) + single window-pass (SW) at the *maximum* evaluation length.
- Robust "packing" tokenizer to create long sequences from short docs (e.g., Wikitext).
- Batched, AMP-aware, works with RNN LMs or any HF-style CausalLM that returns `logits`.

Returned results
----------------
{
  "max_length": int,
  "training_window": int,
  "chunk_size": int,
  "chunk_stride": int,
  "chunk_centers": List[int],
  "chunk_ppl_ttt": List[float],
  "chunk_ppl_sw":  List[float],
  "num_sequences": int,
}

Plot file
---------
chunk_eval_W{W}_T{T}_K{K}_S{S}_{step|final}.png
"""

import os
import math
import logging
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except Exception:
    sns = None

from datasets import load_dataset
import wandb

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perplexity_evaluator")


# =============================================================================
# Evaluator
# =============================================================================
class PerplexityEvaluator:
    def __init__(
        self,
        model,
        tokenizer,
        device: Union[torch.device, str],
        *,
        training_window: int = 64,
        max_seqs: int = 64,
        batch_size: int = 16,          # batch for TTT (full-context) forward
        window_batch_size: int = 256,  # batch for sliding-window window pass
        use_amp: bool = True,
        amp_dtype: Optional[torch.dtype] = None,
        align_prefix_pre_w: bool = True,
        # x-axis mode for plotting/return: 'center' (default) or 'end'
        x_axis: str = "center",
        # chunk config
        chunk_size: Optional[int] = None,   # default: training_window
        chunk_stride: Optional[int] = None, # default: training_window // 4
        # packing config (for building long sequences)
        pack_stride: Optional[int] = None,  # default: target_len (disjoint sequences)
        add_eos_between: bool = True,
        min_doc_chars: int = 50,
    ):
        """
        Args:
            model: HF-style CausalLM returning logits [B, S, V]
            tokenizer: matching tokenizer
            device: torch.device or "cuda[:idx]" or "cpu"
            training_window: W (context cap for baseline)
            max_seqs: number of sequences to evaluate
            batch_size: batch size for full-context forward
            window_batch_size: batch size for window-pass forward
            use_amp: enable autocast on CUDA
            amp_dtype: autocast dtype (bf16 preferred on Ampere+)
            chunk_size: K (chunk length), default W
            chunk_stride: S (chunk stride), default W//4
            pack_stride: stride when slicing packed token stream into length-T sequences
            add_eos_between: insert EOS between docs when packing
            min_doc_chars: filter out tiny paragraphs
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.device_type = self.device.type
        self.model.to(self.device)
        self.model.eval()

        self.W = int(training_window)
        self.max_seqs = int(max_seqs)
        self.batch_size = int(batch_size)
        self.window_batch_size = int(window_batch_size)

        # AMP on CUDA only
        self.use_amp = bool(use_amp and self.device_type == "cuda" and torch.cuda.is_available())
        if amp_dtype is None:
            if self.use_amp:
                idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
                major, _ = torch.cuda.get_device_capability(idx)
                amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
            else:
                amp_dtype = torch.bfloat16
        self.amp_dtype = amp_dtype

        # chunking config
        self.K = int(chunk_size) if chunk_size is not None else self.W
        self.S = int(chunk_stride) if chunk_stride is not None else max(1, self.W // 4)

        # packing config
        self.pack_stride = pack_stride  # may be None → defaults to target_len
        self.add_eos_between = bool(add_eos_between)
        self.min_doc_chars = int(min_doc_chars)

        # exact alignment option: for positions t < W, force SW losses to equal full-context losses
        self.align_prefix_pre_w = bool(align_prefix_pre_w)
        self.x_axis = x_axis if x_axis in ("center", "end") else "center"

        self._token_cache: Optional[List[torch.Tensor]] = None

    # -------------------------------------------------------------------------
    # Data prep / tokenization
    # -------------------------------------------------------------------------
    def _load_texts(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        split: str = "validation",
    ) -> List[str]:
        if dataset_config is not None:
            ds = load_dataset(dataset_name, dataset_config, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
        texts = []
        for ex in ds:
            t = ex.get("text", "")
            if not isinstance(t, str):
                continue
            t = t.strip()
            if len(t) >= self.min_doc_chars:
                texts.append(t)
        if not texts:
            raise RuntimeError("No usable texts found in dataset.")
        return texts

    def build_token_cache_packed(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        target_len: int,
        desired_sequences: int,
    ) -> None:
        """
        Build self._token_cache by packing tokens across many docs into a single stream,
        then slicing sequences of exact `target_len` with stride `pack_stride`.
        Always produces up to `desired_sequences` sequences even if individual docs are short.
        """
        texts = self._load_texts(dataset_name, dataset_config, split="validation")

        # Build one big token stream (optionally insert EOS between docs)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            # fall back to pad or 0 if no EOS configured
            eos_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        stream: List[int] = []
        def _append_ids(ids: List[int]):
            stream.extend(ids)

        # Multiple passes over validation until stream is long enough
        max_passes = 4
        for _ in range(max_passes):
            for txt in texts:
                ids = self.tokenizer.encode(txt, add_special_tokens=False)
                if not ids:
                    continue
                _append_ids(ids)
                if self.add_eos_between:
                    stream.append(eos_id)
            if len(stream) >= target_len * max(1, desired_sequences):
                break

        if len(stream) < target_len:
            raise RuntimeError(f"Packed token stream too short ({len(stream)}) for target_len={target_len}.")

        stride = int(self.pack_stride) if self.pack_stride is not None else target_len
        stride = max(1, stride)

        seqs: List[torch.Tensor] = []
        start = 0
        while start + target_len <= len(stream) and len(seqs) < desired_sequences:
            chunk = stream[start:start + target_len]
            seqs.append(torch.tensor(chunk, dtype=torch.long))
            start += stride

        if len(seqs) < 3:
            raise RuntimeError(
                f"Only built {len(seqs)} sequences (need ≥3). "
                f"Try reducing pack_stride or increasing dataset size."
            )

        self._token_cache = seqs[:desired_sequences]
        logger.info(f"Token cache (packed) built: {len(self._token_cache)} sequences "
                    f"of length {target_len} (pack_stride={stride})")

    # -------------------------------------------------------------------------
    # Core forwards
    # -------------------------------------------------------------------------
    def _full_context_token_nll(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """
        One forward per batch at max length.
        Returns per-token NLL tensor [B, T-1] on CPU (full-context TTT).
        """
        all_ids = torch.stack(sequences, dim=0)  # [B, T]
        B, T = all_ids.shape
        nll = torch.empty((B, T - 1), dtype=torch.float32)

        with torch.inference_mode():
            for b0 in range(0, B, self.batch_size):
                b1 = min(b0 + self.batch_size, B)
                batch = all_ids[b0:b1].to(self.device, non_blocking=True)  # [b, T]
                inp = batch[:, :-1]
                tgt = batch[:, 1:]

                if self.use_amp:
                    with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype):
                        logits = self.model(input_ids=inp).logits  # [b, T-1, V]
                else:
                    logits = self.model(input_ids=inp).logits

                loss_tok = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    reduction="none",
                ).reshape(batch.size(0), -1)  # [b, T-1]

                nll[b0:b1] = loss_tok.detach().cpu()

        return nll  # [B, T-1]

    def _sliding_window_all_positions_losses(self, sequences: List[torch.Tensor], W: int) -> torch.Tensor:
        """
        Exact per-position last-token loss with context limited to at most W tokens.
        Produces a dense tensor [B, T-1] for ALL positions t=1..T-1:

        - For t < W: context length = t (short prefixes). We loop across t (≤ W-1).
        - For t ≥ W: context length = W. We batch all windows with unfold (step=1).
        """
        batch_ids = torch.stack(sequences, dim=0)  # [B, T]
        B, T = batch_ids.shape
        out = torch.empty((B, T - 1), dtype=torch.float32)

        # Early positions (variable context): t = 1..min(W-1, T-1)
        Tmax_early = min(W - 1, T - 1) if W > 1 else 0
        with torch.inference_mode():
            for t in range(1, Tmax_early + 1):
                # predict token at index t from prefix length t
                inp = batch_ids[:, :t].to(self.device, non_blocking=True)     # [B, t]
                tgt = batch_ids[:, t].to(self.device, non_blocking=True)      # [B]

                if self.use_amp:
                    with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype):
                        logits = self.model(input_ids=inp).logits  # [B, t, V]
                else:
                    logits = self.model(input_ids=inp).logits

                last_logits = logits[:, -1, :]       # [B, V]
                loss = F.cross_entropy(last_logits, tgt, reduction="none").detach().cpu()  # [B]
                out[:, t - 1] = loss  # position t → index t-1

        # Regular positions (fixed context W): t = W..T-1
        if T - W > 0 and W >= 1:
            N = T - W
            windows = batch_ids.unfold(dimension=1, size=W + 1, step=1)  # [B, N, W+1]
            inputs = windows[:, :, :-1].reshape(B * N, W)  # [B*N, W]
            targets = windows[:, :, -1].reshape(B * N)     # [B*N]

            losses = torch.empty(B * N, dtype=torch.float32)
            with torch.inference_mode():
                for s in range(0, B * N, self.window_batch_size):
                    e = min(s + self.window_batch_size, B * N)
                    inp = inputs[s:e].to(self.device, non_blocking=True)
                    tgt = targets[s:e].to(self.device, non_blocking=True)

                    if self.use_amp:
                        with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype):
                            out_logits = self.model(input_ids=inp).logits  # [b, W, V]
                    else:
                        out_logits = self.model(input_ids=inp).logits

                    last_logits = out_logits[:, -1, :]
                    loss = F.cross_entropy(last_logits, tgt, reduction="none").detach().cpu()
                    losses[s:e] = loss

            out[:, W - 1:] = losses.view(B, N)  # fill positions t=W..T-1 (index t-1 starts at W-1)

        return out  # [B, T-1]

    # -------------------------------------------------------------------------
    # Chunk aggregation (same for TTT and SW)
    # -------------------------------------------------------------------------
    @staticmethod
    def _chunk_positions_and_means(
        nll_tok: torch.Tensor,
        K: int,
        S: int,
        *,
        clip_prefix_at: Optional[int] = None,
        clip_mode: str = "center",  # 'center' or 'end'
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        From per-token NLL [B, T-1], compute per-chunk median-of-mean losses:
        - chunks are [s : s+K) over token-loss indices (0..T-2)
        - stride S
        Returns (centers, ends, ppls).
        Clipping rules:
        - clip_mode='center': if center < clip_at and end > clip_at → end := clip_at
        - clip_mode='end':    if end > clip_at → end := clip_at (ensures chunks with end≤W only use ≤W losses)
        """
        B, Tm1 = nll_tok.shape
        if K < 1:
            raise ValueError("chunk_size (K) must be >= 1")
        starts = list(range(0, max(0, Tm1 - K + 1), max(1, S)))
        if not starts:
            return [], [], []
        cumsum = torch.cumsum(nll_tok, dim=1)  # [B, T-1]
        ppls, centers, ends = [], [], []
        for s in starts:
            e = s + K
            length = K
            if clip_prefix_at is not None and e > clip_prefix_at:
                if clip_mode == "center":
                    center = s + (K // 2)
                    if center < clip_prefix_at:
                        e = clip_prefix_at
                        length = max(1, e - s)
                # In 'end' mode, we don't clip post-W chunks; pre-W ends (<=W) never straddle by construction.

            sums = cumsum[:, e - 1] - (cumsum[:, s - 1] if s > 0 else 0)
            means = sums / length                  # [B]
            med = torch.median(means).item()
            ppls.append(float(math.exp(med)))
            centers.append(s + K // 2)
            ends.append(e)
        return centers, ends, ppls

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def evaluate_chunks_only(
        self,
        *,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-v1",
        max_length: Optional[int] = None,   # None => 10×W
        use_packing: bool = True,
    ) -> Dict:
        """
        Evaluate ChunkPPL_TTT and ChunkPPL_SW (same chunking).
        - TTT: one full-context forward at length T.
        - SW:  one window pass to build per-position losses with context ≤ W.
        - Data: by default, pack tokens across many docs to ensure long sequences.
        """
        W = self.W
        K = self.K
        S = self.S
        T = int(W * 200) if max_length is None else int(max_length)

        # 1) Build sequences of exact length T
        if use_packing:
            self.build_token_cache_packed(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                target_len=T,
                desired_sequences=self.max_seqs,
            )
            sequences = self._token_cache
        else:
            # Fallback: naive per-doc truncation (will fail if docs are short)
            texts = self._load_texts(dataset_name, dataset_config, split="validation")
            enc = self.tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                max_length=T,
                return_attention_mask=False,
            )
            seqs = [torch.tensor(x, dtype=torch.long) for x in enc["input_ids"] if len(x) >= T]
            sequences = seqs[: self.max_seqs]

        if not sequences or len(sequences) < 3:
            raise RuntimeError(f"Not enough sequences of length {T} to evaluate (need ≥3).")

        # 2) TTT (full-context) per-token NLL → ChunkPPL_TTT
        nll_full = self._full_context_token_nll(sequences)               # [B, T-1]
        # We will compute chunk means with optional prefix clipping for strict alignment
        # Use clip boundary at W so pre-W chunks (center < W) never include tokens at or beyond W.
        clip_at = self.W if self.align_prefix_pre_w else None
        clip_mode = "end" if self.x_axis == "end" else "center"
        centers_full, ends_full, chunk_ppl_ttt = self._chunk_positions_and_means(
            nll_full, K, S, clip_prefix_at=clip_at, clip_mode=clip_mode
        )

        # 3) SW baseline per-position NLL (context≤W) → ChunkPPL_SW
        nll_sw = self._sliding_window_all_positions_losses(sequences, W) # [B, T-1]

        # Optional: enforce exact alignment for prefix region t < W by reusing the full-context losses
        if self.align_prefix_pre_w and nll_sw.numel() > 0:
            Tmax_early = min(W - 1, nll_sw.shape[1]) if W > 1 else 0
            if Tmax_early > 0:
                nll_sw[:, :Tmax_early] = nll_full[:, :Tmax_early]

        _, ends_sw, chunk_ppl_sw  = self._chunk_positions_and_means(
            nll_sw,   K, S, clip_prefix_at=clip_at, clip_mode=clip_mode
        )

        # Choose x-axis series
        chunk_x = ends_full if self.x_axis == "end" else centers_full

        # Back-compat: also expose centers and ends explicitly
        chunk_centers = centers_full
        chunk_ends = ends_sw  # ends are same indexing across TTT/SW by construction

        results = {
            "max_length": T,
            "training_window": W,
            "chunk_size": K,
            "chunk_stride": S,
            "x_mode": self.x_axis,                   # 'center' or 'end'
            "chunk_x": chunk_x,                      # x-values used for plotting
            "chunk_centers": chunk_centers,          # centers (for compatibility)
            "chunk_ends": chunk_ends,                # right-edge positions
            "chunk_ppl_ttt": chunk_ppl_ttt,          # full-context chunk ppl
            "chunk_ppl_sw": chunk_ppl_sw,            # sliding-window baseline chunk ppl
            "num_sequences": len(sequences),
        }
        logger.info("✅ Chunk-only evaluation complete.")
        return results

    # -------------------------------------------------------------------------
    # Plot + W&B
    # -------------------------------------------------------------------------
    def plot_chunks(self, res: Dict, save_path: Optional[str] = None, show: bool = True) -> str:
        xvals = res.get("chunk_x") or res["chunk_centers"]
        ttt = res["chunk_ppl_ttt"]
        sw  = res["chunk_ppl_sw"]
        W   = res["training_window"]
        T   = res["max_length"]
        K   = res["chunk_size"]
        S   = res["chunk_stride"]
        x_mode = res.get("x_mode", "center")

        if not xvals or not ttt or not sw:
            raise ValueError("Missing chunk results to plot.")

        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        # Clean lines only (no markers) to avoid visual artifacts near W
        ax.plot(xvals, ttt, linewidth=2, label="ChunkPPL (TTT, full-context)")
        ax.plot(xvals, sw,  linewidth=2, label=f"ChunkPPL (SW, context≤{W})")

        # Guides
        ax.axvline(W, color="red", linestyle="--", alpha=0.6, label=f"Training Window ({W})")
        for m in [2, 5, 10]:
            x = W * m
            if x <= T:
                ax.axvline(x, color="orange", linestyle=":", alpha=0.4)

        xlabel = "Token position (chunk end)" if x_mode == "end" else "Token position (chunk center)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Perplexity")
        ax.set_title(f"ChunkPPL: Full-context vs Sliding-window (K={K}, stride={S})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        if save_path is None:
            save_path = f"chunk_eval_W{W}_T{T}_K{K}_S{S}.png"
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"📊 Plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return save_path

    def log_to_wandb(self, res: Dict, plot_path: str, step: Optional[int] = None):
        if not wandb.run:
            logger.warning("No active wandb run, skipping logging")
            return

        W = res["training_window"]
        K = res["chunk_size"]
        S = res["chunk_stride"]
        centers = res["chunk_centers"]
        ttt = res["chunk_ppl_ttt"]
        sw  = res["chunk_ppl_sw"]

        wandb.log({"chunk_eval_plot": wandb.Image(plot_path)}, step=step)

        # Log a few anchor points near 1x/5x/10x windows (by center)
        def closest(x):
            if not centers:
                return None
            return int(np.argmin([abs(c - x) for c in centers]))

        for r in [1, 5, 10]:
            x = W * r
            i = closest(x)
            if i is not None:
                wandb.log({
                    f"eval/chunk_ttt_ppl_at_{r}xW": ttt[i],
                    f"eval/chunk_sw_ppl_at_{r}xW":  sw[i],
                }, step=step)

        # Summary stats
        if ttt and sw:
            wandb.log({
                "eval/chunk_ttt_min": min(ttt),
                "eval/chunk_ttt_max": max(ttt),
                "eval/chunk_sw_min":  min(sw),
                "eval/chunk_sw_max":  max(sw),
                "eval/training_window": W,
                "eval/chunk_size": K,
                "eval/chunk_stride": S,
            }, step=step)


# =============================================================================
# Convenience wrapper
# =============================================================================
def evaluate_model_perplexity(
    model,
    tokenizer,
    device: Union[torch.device, str],
    *,
    training_window: int = 64,
    max_seqs: int = 64,
    batch_size: int = 16,
    window_batch_size: int = 256,
    use_amp: bool = True,
    amp_dtype: Optional[torch.dtype] = None,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-v1",
    max_length: Optional[int] = None,    # if None => 10×W
    chunk_size: Optional[int] = None,    # default W
    chunk_stride: Optional[int] = None,  # default W//4
    # packing knobs
    pack_stride: Optional[int] = None,   # default target_len (disjoint sequences)
    add_eos_between: bool = True,
    min_doc_chars: int = 50,
    # I/O
    output_dir: str = "./",
    step: Optional[str] = None,
    log_wandb: bool = False,
    use_packing: bool = True,
    align_prefix_pre_w: bool = True,
    x_axis: str = "center",
) -> Dict:
    """
    Chunk-only evaluation:
      - ChunkPPL_TTT (full-context, single pass)
      - ChunkPPL_SW  (sliding-window baseline, one window pass)
    """
    evaluator = PerplexityEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        training_window=training_window,
        max_seqs=max_seqs,
        batch_size=batch_size,
        window_batch_size=window_batch_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        align_prefix_pre_w=align_prefix_pre_w,
    x_axis=x_axis,
        chunk_size=chunk_size,
        chunk_stride=chunk_stride,
        pack_stride=pack_stride,
        add_eos_between=add_eos_between,
        min_doc_chars=min_doc_chars,
    )

    results = evaluator.evaluate_chunks_only(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_length=max_length,
        use_packing=use_packing,
    )

    # Plot
    plot_name = f"chunk_eval_W{results['training_window']}_T{results['max_length']}_K{results['chunk_size']}_S{results['chunk_stride']}_{step if step is not None else 'final'}.png"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, plot_name)
    evaluator.plot_chunks(results, save_path=plot_path, show=False)

    # Log
    if log_wandb:
        evaluator.log_to_wandb(results, plot_path, step)

    return results


# =============================================================================
# Example (commented)
# =============================================================================
# if __name__ == "__main__":
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     name = "gpt2"  # replace with your RNN LM if it returns logits [B,S,V]
#     tok = AutoTokenizer.from_pretrained(name)
#     mdl = AutoModelForCausalLM.from_pretrained(name).to(dev)
#
#     res = evaluate_model_perplexity(
#         mdl, tok, dev,
#         training_window=64,
#         max_seqs=64,
#         batch_size=16,
#         window_batch_size=256,
#         use_amp=True,
#         dataset_name="wikitext",
#         dataset_config="wikitext-2-v1",
#         max_length=None,      # None => 10×W
#         chunk_size=None,      # None => W
#         chunk_stride=None,    # None => W//4
#         pack_stride=None,     # None => disjoint sequences of length T
#         output_dir="./outputs",
#         log_wandb=False,
#         use_packing=True,
#     )
#     print({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()})
