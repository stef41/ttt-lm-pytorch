#!/usr/bin/env python3
"""
Curriculum training for studying undertraining's effect on length generalization.

Takes a 125M TTT model from initialization to convergence on W=64, evaluating
length generalization at T=20×W exactly 30 times during training to track
how length generalization evolves as the model learns.

Usage:
  python train_curriculum_length_gen.py \\
    --output-dir curriculum_exp \\
    --max-train-steps 15000 \\
    --num-eval-checkpoints 30 \\
    --eval-max-length 1280
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# Add the current directory to Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS, TTTCache
from perplexity_evaluator import evaluate_model_perplexity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    # Model & data
    p.add_argument("--model_size", type=str, default="125m", choices=["125m", "350m", "760m", "1b"])
    p.add_argument("--ttt_layer_type", type=str, default="linear", choices=["linear", "mlp"])
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--dataset_subset_size", type=int, default=-1, help="-1 for full dataset")
    
    # Training window & length generalization
    p.add_argument("--max_seq_length", type=int, default=64, help="Training window W")
    p.add_argument("--eval_max_length", type=int, default=1280, help="Eval length T (e.g., 20×W)")
    
    # Training schedule
    p.add_argument("--max_train_steps", type=int, default=15000, help="Total training steps")
    p.add_argument("--num_eval_checkpoints", type=int, default=30, help="Number of eval checkpoints")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=50)
    
    # Optimizer
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    
    # Evaluation
    p.add_argument("--eval_max_seqs", type=int, default=32, help="Sequences for length gen eval")
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--eval_window_batch_size", type=int, default=128)
    p.add_argument("--eval_chunk_size", type=int, default=16)
    p.add_argument("--eval_chunk_stride", type=int, default=16)
    
    # System
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--output_dir", type=str, default="./curriculum_exp")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--state_passing", action="store_true", default=True)
    p.add_argument("--no_state_passing", dest="state_passing", action="store_false")
    p.add_argument("--state_reset_interval", type=int, default=0, help="0=never reset")
    
    return p.parse_args()


def get_model_config(args):
    config_dict = TTT_STANDARD_CONFIGS[args.model_size].copy()
    config_dict.update({
        "max_position_embeddings": args.max_seq_length,
        "ttt_layer_type": args.ttt_layer_type,
        "use_cache": False,
        "state_passing": args.state_passing,
    })
    return TTTConfig(**config_dict)


def create_data_collator(tokenizer):
    def data_collator(features):
        batch = {}
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch["input_ids"] = []
        batch["attention_mask"] = []
        batch["labels"] = []
        
        for f in features:
            input_ids = f["input_ids"]
            attention_mask = [1] * len(input_ids)
            labels = f["labels"]
            
            pad_length = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length
            labels += [-100] * pad_length
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        
        return {k: torch.tensor(v) for k, v in batch.items()}
    return data_collator


def run_length_gen_eval(model, tokenizer, args, accelerator, checkpoint_idx, step):
    """Run length generalization evaluation and save results."""
    logger.info(f"📊 Checkpoint {checkpoint_idx}/30 (step {step}): Running length gen eval...")
    
    try:
        res = evaluate_model_perplexity(
            model=accelerator.unwrap_model(model),
            tokenizer=tokenizer,
            device=accelerator.device,
            training_window=args.max_seq_length,
            max_seqs=args.eval_max_seqs,
            batch_size=args.eval_batch_size,
            window_batch_size=args.eval_window_batch_size,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            max_length=args.eval_max_length,
            chunk_size=args.eval_chunk_size,
            chunk_stride=args.eval_chunk_stride,
            output_dir=os.path.join(args.output_dir, f"checkpoint_{checkpoint_idx:03d}"),
            step=f"ckpt{checkpoint_idx}",
            log_wandb=False,
            use_packing=True,
            align_prefix_pre_w=True,
            x_axis="end",
        )
        
        # Save summary stats
        x = res.get("chunk_x") or res["chunk_centers"]
        ttt_ppl = res["chunk_ppl_ttt"]
        sw_ppl = res["chunk_ppl_sw"]
        W = res["training_window"]
        
        def closest_idx(target):
            return min(range(len(x)), key=lambda i: abs(x[i] - target)) if x else None
        
        summary = {
            "checkpoint_idx": checkpoint_idx,
            "step": step,
            "training_window": W,
            "eval_max_length": res["max_length"],
            "chunk_size": res["chunk_size"],
            "chunk_stride": res["chunk_stride"],
        }
        
        # Sample key positions
        for r in [1, 2, 5, 10, 20]:
            target = W * r
            idx = closest_idx(target)
            if idx is not None and idx < len(ttt_ppl):
                summary[f"ttt_ppl_at_{r}xW"] = ttt_ppl[idx]
                summary[f"sw_ppl_at_{r}xW"] = sw_ppl[idx]
                summary[f"x_pos_at_{r}xW"] = x[idx]
        
        # Overall stats
        if ttt_ppl and sw_ppl:
            summary["ttt_ppl_mean"] = float(np.mean(ttt_ppl))
            summary["ttt_ppl_min"] = float(np.min(ttt_ppl))
            summary["ttt_ppl_max"] = float(np.max(ttt_ppl))
            summary["sw_ppl_mean"] = float(np.mean(sw_ppl))
            summary["sw_ppl_min"] = float(np.min(sw_ppl))
            summary["sw_ppl_max"] = float(np.max(sw_ppl))
        
        summary_path = os.path.join(args.output_dir, f"checkpoint_{checkpoint_idx:03d}", "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Checkpoint {checkpoint_idx}: Eval complete. Summary saved to {summary_path}")
        
        # Quick peek at key metrics
        if "ttt_ppl_at_1xW" in summary and "ttt_ppl_at_10xW" in summary:
            logger.info(f"   @1xW: TTT={summary['ttt_ppl_at_1xW']:.1f}, SW={summary['sw_ppl_at_1xW']:.1f}")
            logger.info(f"   @10xW: TTT={summary['ttt_ppl_at_10xW']:.1f}, SW={summary['sw_ppl_at_10xW']:.1f}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Checkpoint {checkpoint_idx}: Eval failed: {e}")
        return None


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
        kwargs_handlers=[ddp_kwargs],
    )
    
    logger.info("🚀 Curriculum Training for Length Generalization")
    logger.info(f"  Model: {args.model_size}, W={args.max_seq_length}, eval T={args.eval_max_length}")
    logger.info(f"  Total steps: {args.max_train_steps}, Eval checkpoints: {args.num_eval_checkpoints}")
    logger.info(f"  Device: {accelerator.device}, Mixed precision: {args.mixed_precision}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    config = get_model_config(args)
    config.vocab_size = len(tokenizer)
    model = TTTForCausalLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {total_params:,}")
    
    # Dataset
    logger.info("📁 Loading dataset...")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    train_dataset = dataset["train"]
    if args.dataset_subset_size > 0:
        train_dataset = train_dataset.select(range(min(args.dataset_subset_size, len(train_dataset))))
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=args.max_seq_length,
            return_overflowing_tokens=True,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) >= 10)
    logger.info(f"  Training examples: {len(train_dataset)}")
    
    # DataLoader
    data_collator = create_data_collator(tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        foreach=True,
    )
    warmup_steps = max(1, int(args.warmup_ratio * args.max_train_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    # Compute eval checkpoints (equally spaced + initial + final)
    # We want 30 total: step 0, then 28 intermediate, then final
    if args.num_eval_checkpoints < 2:
        raise ValueError("num_eval_checkpoints must be >= 2 (init + final)")
    
    eval_steps = [0]  # Always eval at init
    if args.num_eval_checkpoints > 2:
        # Intermediate checkpoints (skip 0 and max_train_steps, add them manually)
        intermediate = args.num_eval_checkpoints - 2
        step_interval = args.max_train_steps // (intermediate + 1)
        for i in range(1, intermediate + 1):
            eval_steps.append(i * step_interval)
    eval_steps.append(args.max_train_steps)  # Final
    
    # Remove duplicates and sort
    eval_steps = sorted(set(eval_steps))
    logger.info(f"📅 Evaluation schedule: {len(eval_steps)} checkpoints at steps {eval_steps[:5]}...{eval_steps[-3:]}")
    
    # Save metadata
    os.makedirs(args.output_dir, exist_ok=True)
    metadata = {
        "model_size": args.model_size,
        "training_window": args.max_seq_length,
        "eval_max_length": args.eval_max_length,
        "max_train_steps": args.max_train_steps,
        "num_eval_checkpoints": len(eval_steps),
        "eval_steps": eval_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": warmup_steps,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "state_passing": args.state_passing,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Training loop
    logger.info("🏃 Starting curriculum training...")
    model.train()
    
    # TTT cache for state passing
    ttt_cache = None
    if config.state_passing:
        logger.info("🔗 Initializing TTT cache for state passing...")
        ttt_cache = TTTCache(accelerator.unwrap_model(model).model, args.per_device_train_batch_size)
    
    start_time = time.time()
    total_loss = 0.0
    tokens_processed = 0
    eval_checkpoint_idx = 0
    all_summaries = []
    
    # Eval at initialization (step 0)
    if accelerator.is_main_process:
        summary = run_length_gen_eval(model, tokenizer, args, accelerator, eval_checkpoint_idx, step=0)
        if summary:
            all_summaries.append(summary)
        eval_checkpoint_idx += 1
    
    step = 0
    data_iter = iter(train_dataloader)
    
    while step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        
        step_start = time.time()
        
        # Handle cache batch size mismatch
        if config.state_passing:
            current_bs = batch['input_ids'].shape[0]
            try:
                if ttt_cache is not None:
                    first_key = next(iter(ttt_cache.ttt_params_dict))
                    first_layer_idx = next(iter(ttt_cache.ttt_params_dict[first_key]))
                    cache_bs = ttt_cache.ttt_params_dict[first_key][first_layer_idx].shape[0]
                    if cache_bs != current_bs:
                        ttt_cache = TTTCache(accelerator.unwrap_model(model).model, current_bs)
            except (StopIteration, KeyError):
                ttt_cache = TTTCache(accelerator.unwrap_model(model).model, current_bs)
        
        with accelerator.accumulate(model):
            forward_kwargs = batch.copy()
            if config.state_passing and ttt_cache is not None:
                forward_kwargs['cache_params'] = ttt_cache
            
            outputs = model(**forward_kwargs)
            loss = outputs.loss
            total_loss += loss.detach().float()
            tokens_processed += batch['input_ids'].numel()
            
            accelerator.backward(loss)
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        step_time = time.time() - step_start
        tokens_per_sec = batch['input_ids'].numel() / step_time
        
        # Periodically reset cache
        if (config.state_passing and ttt_cache is not None and 
            args.state_reset_interval > 0 and 
            (step + 1) % args.state_reset_interval == 0):
            current_bs = batch['input_ids'].shape[0]
            ttt_cache = TTTCache(accelerator.unwrap_model(model).model, current_bs)
        
        # Logging
        if args.logging_steps > 0 and step % args.logging_steps == 0:
            avg_loss = total_loss / max(1, args.logging_steps)
            try:
                train_ppl = float(math.exp(avg_loss)) if avg_loss < 50 else float('inf')
            except Exception:
                train_ppl = float('nan')
            
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            
            logger.info(
                f"Step {step:5d}/{args.max_train_steps}: "
                f"loss={avg_loss:.4f} (ppl~{train_ppl:.1f}), "
                f"lr={current_lr:.2e}, "
                f"tokens/sec={tokens_per_sec:.0f}"
            )
            total_loss = 0.0
        
        step += 1
        
        # Check for eval checkpoint
        if step in eval_steps and accelerator.is_main_process:
            # Switch to eval mode
            model.eval()
            summary = run_length_gen_eval(model, tokenizer, args, accelerator, eval_checkpoint_idx, step)
            if summary:
                all_summaries.append(summary)
            eval_checkpoint_idx += 1
            # Back to train mode
            model.train()
    
    # Final checkpoint if not already done
    if args.max_train_steps not in eval_steps and accelerator.is_main_process:
        model.eval()
        summary = run_length_gen_eval(model, tokenizer, args, accelerator, eval_checkpoint_idx, args.max_train_steps)
        if summary:
            all_summaries.append(summary)
    
    # Save all summaries
    if accelerator.is_main_process:
        all_summaries_path = os.path.join(args.output_dir, "all_summaries.json")
        with open(all_summaries_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        logger.info(f"📊 All summaries saved to {all_summaries_path}")
    
    # Save final model
    if accelerator.is_main_process:
        final_model_dir = os.path.join(args.output_dir, "final_model")
        accelerator.unwrap_model(model).save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        logger.info(f"💾 Final model saved to {final_model_dir}")
    
    total_time = time.time() - start_time
    logger.info(f"\n✅ Curriculum training complete!")
    logger.info(f"  Total time: {total_time/60:.1f}m")
    logger.info(f"  Total steps: {step}")
    logger.info(f"  Eval checkpoints: {len(all_summaries)}")


if __name__ == "__main__":
    main()
