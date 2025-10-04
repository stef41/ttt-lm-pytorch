#!/usr/bin/env python3
"""
GPU-optimized training script for TTT models with explicit GPU usage and monitoring.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# Add the current directory to Python path to import ttt module
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, set_seed
from accelerate import Accelerator
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS, TTTCache
from perplexity_evaluator import evaluate_model_perplexity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_subset_size", type=int, default=-1,
                        help="If >0, use only the first N examples for training; -1 uses full dataset")
    parser.add_argument("--model_size", type=str, default="125m", choices=["125m", "350m", "760m", "1b"])
    parser.add_argument("--ttt_layer_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_seq_length", type=int, default=64)  # Set to 64 tokens for all experiments
    parser.add_argument("--max_train_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--output_dir", type=str, default="./gpu_test_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state_passing", action="store_true", default=True, 
                        help="Enable state passing between batches for continuous learning")
    parser.add_argument("--no_state_passing", dest="state_passing", action="store_false",
                        help="Disable state passing between batches")
    parser.add_argument("--state_reset_interval", type=int, default=100,
                        help="Reset TTT state cache every N steps (0 = never reset)")
    parser.add_argument("--wandb_project", type=str, default="ttt-gpu-optimized", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb_offline", action="store_true", help="Force W&B offline mode")
    # Evaluation controls
    parser.add_argument("--eval_skip", action="store_true", help="Skip perplexity evaluation at the end")
    parser.add_argument(
        "--eval_max_length_factor", type=int, default=20,
        help="Evaluation sequence length is factor * max_seq_length (set 0 to skip)"
    )
    parser.add_argument("--eval_max_seqs", type=int, default=16, help="Max sequences to evaluate")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for full-context eval")
    parser.add_argument("--eval_window_batch_size", type=int, default=64, help="Batch size for window eval")
    parser.add_argument("--eval_use_packing", action="store_true", default=True, help="Pack tokens for eval")
    parser.add_argument("--eval_dataset_name", type=str, default="wikitext", help="Eval dataset name")
    parser.add_argument("--eval_dataset_config", type=str, default="wikitext-2-raw-v1", help="Eval dataset config")
    # Optimizer & scheduler knobs
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2 (lower than 0.999 for faster adapt)")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm (0 = disable)")
    parser.add_argument("--warmup_ratio", type=float, default=0.02, help="Linear/cosine warmup ratio of total steps")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine"], help="LR scheduler type")
    # Evaluation knobs
    parser.add_argument("--eval_chunk_size", type=int, default=16, help="Chunk size K for evaluation")
    parser.add_argument("--eval_chunk_stride", type=int, default=16, help="Chunk stride S for evaluation")
    parser.add_argument("--eval_x_axis", type=str, default="end", choices=["center", "end"], help="X-axis for plots")
    parser.add_argument("--eval_align_prefix", action="store_true", default=True, help="Force pre-W alignment")
    # Early stopping on plateau
    parser.add_argument("--early_stop_patience", type=int, default=0, help="Stop after N logging windows with < min_delta improvement (0=disable)")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.01, help="Minimum absolute improvement in avg loss to reset patience")
    parser.add_argument("--min_train_steps", type=int, default=200, help="Minimum number of steps before early stopping can trigger")
    parser.add_argument("--save_best", action="store_true", help="Save best checkpoint as best_model during training")
    return parser.parse_args()

def monitor_gpu_usage():
    """Monitor and log GPU usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak"
    return "No GPU available"

def get_model_config(args):
    config_dict = TTT_STANDARD_CONFIGS[args.model_size].copy()
    config_dict.update({
        "max_position_embeddings": args.max_seq_length,
        "ttt_layer_type": args.ttt_layer_type,
        "use_cache": False,
        "state_passing": args.state_passing,  # Pass the state_passing argument to config
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

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Initialize accelerator with explicit GPU settings
    # Set DDP kwargs for state passing compatibility
    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
        kwargs_handlers=[ddp_kwargs],
    )
    
    # Initialize wandb
    if not args.no_wandb and accelerator.is_main_process:
        wandb_mode = "offline" if args.wandb_offline else "online"
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"gpu-{args.model_size}-bs{args.per_device_train_batch_size}x{args.gradient_accumulation_steps}-lr{args.learning_rate}",
            mode=wandb_mode,  # Online by default, offline if specified
            config={
                "model_size": args.model_size,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "ttt_layer_type": args.ttt_layer_type,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "max_seq_length": args.max_seq_length,
                "max_train_steps": args.max_train_steps,
                "mixed_precision": args.mixed_precision,
                "state_passing": args.state_passing,
                "state_reset_interval": args.state_reset_interval,
                "num_processes": accelerator.num_processes,
                "training_type": "gpu_optimized",
                "seed": args.seed
            }
        )
    
    logger.info(f"🚀 GPU Training Setup:")
    logger.info(f"  Device: {accelerator.device}")
    logger.info(f"  Num processes: {accelerator.num_processes}")
    logger.info(f"  Mixed precision: {args.mixed_precision}")
    logger.info(f"  Model size: {args.model_size}")
    logger.info(f"  Batch size per device: {args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  W&B logging: {'enabled' if not args.no_wandb else 'disabled'}")
    if not args.no_wandb:
        logger.info(f"  W&B mode: {'offline' if args.wandb_offline else 'online'}")
    logger.info(f"  {monitor_gpu_usage()}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    config = get_model_config(args)
    config.vocab_size = len(tokenizer)
    
    logger.info(f"📋 Model Configuration:")
    logger.info(f"  Hidden size: {config.hidden_size}")
    logger.info(f"  Layers: {config.num_hidden_layers}")
    logger.info(f"  Attention heads: {config.num_attention_heads}")
    logger.info(f"  Vocab size: {config.vocab_size}")
    logger.info(f"  State passing: {config.state_passing}")
    logger.info(f"  TTT layer type: {config.ttt_layer_type}")
    
    model = TTTForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  {monitor_gpu_usage()}")
    
    # Log model info to wandb
    if not args.no_wandb and accelerator.is_main_process:
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/hidden_size": config.hidden_size,
            "model/num_layers": config.num_hidden_layers,
            "model/num_heads": config.num_attention_heads,
            "model/vocab_size": config.vocab_size,
            "model/max_seq_length": config.max_position_embeddings,
        })
    
    # Load and prepare dataset
    logger.info("📁 Loading dataset...")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    train_dataset = dataset["train"]
    if args.dataset_subset_size and args.dataset_subset_size > 0:
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
    
    # Create data loader
    data_collator = create_data_collator(tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        foreach=True,
    )
    warmup_steps = max(1, int(args.warmup_ratio * args.max_train_steps)) if args.warmup_ratio and args.max_train_steps > 0 else 0
    if args.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    
    # Prepare everything with accelerator (this moves to GPU)
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    logger.info(f"🎯 After prepare: {monitor_gpu_usage()}")
    
    # Training loop with detailed monitoring
    logger.info("🏃 Starting training...")
    model.train()
    
    # Initialize TTT cache for state passing
    ttt_cache = None
    if config.state_passing:
        logger.info("🔗 Initializing TTT cache for state passing...")
        # Create cache with batch size - use model.model to access the base TTTModel
        ttt_cache = TTTCache(accelerator.unwrap_model(model).model, args.per_device_train_batch_size)
        logger.info(f"   Cache created for batch size: {args.per_device_train_batch_size}")
    
    start_time = time.time()
    total_loss = 0.0
    tokens_processed = 0
    
    step = 0
    data_iter = iter(train_dataloader)
    # Early stopping state
    best_avg_loss = float("inf")
    no_improve_windows = 0
    while step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        step_start = time.time()
        
        # Verify data is on GPU
        if step == 0:
            logger.info(f"  Batch input_ids device: {batch['input_ids'].device}")
            logger.info(f"  Batch shape: {batch['input_ids'].shape}")
        
        # Ensure TTT cache batch size matches current batch size (drop_last may be False)
        if config.state_passing:
            current_bs = batch['input_ids'].shape[0]
            try:
                # Infer cache batch size from first param tensor
                cache_bs = None
                if ttt_cache is not None:
                    first_key = next(iter(ttt_cache.ttt_params_dict))
                    # pick the first layer present
                    first_layer_idx = next(iter(ttt_cache.ttt_params_dict[first_key]))
                    cache_bs = ttt_cache.ttt_params_dict[first_key][first_layer_idx].shape[0]
                if ttt_cache is None or cache_bs != current_bs:
                    ttt_cache = TTTCache(accelerator.unwrap_model(model).model, current_bs)
                    logger.info(f"🔁 Reinitialized TTT cache for batch size: {current_bs}")
            except StopIteration:
                # Fallback: if cache dicts are empty for some reason, recreate with current batch size
                ttt_cache = TTTCache(accelerator.unwrap_model(model).model, current_bs)
                logger.info(f"🔁 Reinitialized empty TTT cache for batch size: {current_bs}")

        with accelerator.accumulate(model):
            # Prepare forward pass arguments
            forward_kwargs = batch.copy()
            
            # Add cache for state passing if enabled
            if config.state_passing and ttt_cache is not None:
                forward_kwargs['cache_params'] = ttt_cache
                
            outputs = model(**forward_kwargs)
            loss = outputs.loss
            total_loss += loss.detach().float()
            
            # Count tokens processed
            tokens_processed += batch['input_ids'].numel()
            
            accelerator.backward(loss)
            # Gradient clipping
            if args.max_grad_norm and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        step_time = time.time() - step_start
        tokens_per_sec = batch['input_ids'].numel() / step_time
        
        # Periodically reset TTT cache to prevent memory issues and ensure stability
        if (config.state_passing and ttt_cache is not None and 
            args.state_reset_interval > 0 and 
            (step + 1) % args.state_reset_interval == 0):
            logger.info(f"🔄 Resetting TTT cache at step {step + 1}")
            # Reset with current batch size, not the configured per-device batch size
            current_bs = batch['input_ids'].shape[0]
            ttt_cache = TTTCache(accelerator.unwrap_model(model).model, current_bs)
        
        if args.logging_steps > 0 and step % args.logging_steps == 0:
            avg_loss = total_loss / max(1, args.logging_steps)
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            # Convert to train perplexity (best-effort; may be large during early training)
            try:
                train_ppl = float(math.exp(avg_loss)) if avg_loss < 50 else float('inf')
            except Exception:
                train_ppl = float('nan')
            
            state_info = ""
            if config.state_passing:
                state_info = f", state_passing=ON"
                if args.state_reset_interval > 0:
                    state_info += f" (reset_every={args.state_reset_interval})"
                else:
                    state_info += " (no_reset)"
            else:
                state_info = ", state_passing=OFF"
                
            logger.info(f"Step {step:3d}: "
                       f"loss={avg_loss:.4f} (ppl~{train_ppl:.1f}), "
                       f"lr={current_lr:.2e}, "
                       f"tokens/sec={tokens_per_sec:.0f}, "
                       f"time={step_time:.2f}s{state_info}")
            logger.info(f"         {monitor_gpu_usage()}")
            
            # Wandb logging
            if not args.no_wandb and accelerator.is_main_process:
                # Extract GPU memory info
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
                
                wandb.log({
                    "train/loss": avg_loss,
                    "train/ppl": train_ppl,
                    "train/learning_rate": current_lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step_time": step_time,
                    "train/tokens_processed": tokens_processed,
                    "train/step": step,
                    "train/epoch": step / len(train_dataloader),
                    "system/gpu_memory_allocated_gb": gpu_mem_allocated,
                    "system/gpu_memory_reserved_gb": gpu_mem_reserved,
                    "system/elapsed_time": elapsed,
                }, step=step)
            
            total_loss = 0.0
            # Early stopping check (uses avg_loss computed over the last logging window)
            if args.early_stop_patience > 0:
                improved = (best_avg_loss - avg_loss) >= args.early_stop_min_delta
                if improved:
                    best_avg_loss = avg_loss
                    no_improve_windows = 0
                    # Optionally save best checkpoint
                    if args.save_best and accelerator.is_main_process:
                        os.makedirs(args.output_dir, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(model)
                        best_dir = os.path.join(args.output_dir, "best_model")
                        unwrapped.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        logger.info(f"💾 Saved new best checkpoint to {best_dir} (avg_loss={avg_loss:.4f})")
                else:
                    no_improve_windows += 1
                    if step >= args.min_train_steps and no_improve_windows >= args.early_stop_patience:
                        logger.info(
                            f"⏹️ Early stopping triggered at step {step}: no improvement for {no_improve_windows} windows (best={best_avg_loss:.4f}, current={avg_loss:.4f})"
                        )
                        break
        step += 1

    # Final statistics
    total_time = time.time() - start_time
    avg_tokens_per_sec = tokens_processed / total_time
    
    logger.info(f"\n📊 Training Complete:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Total tokens: {tokens_processed:,}")
    logger.info(f"  Average tokens/sec: {avg_tokens_per_sec:.0f}")
    logger.info(f"  Final {monitor_gpu_usage()}")
    
    # Final wandb logging
    if not args.no_wandb and accelerator.is_main_process:
        wandb.log({
            "summary/total_training_time": total_time,
            "summary/total_tokens_processed": tokens_processed,
            "summary/avg_tokens_per_sec": avg_tokens_per_sec,
            "summary/steps_completed": step + 1,
            "summary/final_gpu_memory_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        })
    
    # Test inference speed
    logger.info("\n🧪 Testing inference speed...")
    model.eval()
    
    test_input = tokenizer("The future of AI is", return_tensors="pt")
    test_input = {k: v.to(accelerator.device) for k, v in test_input.items()}
    
    inference_start = time.time()
    with torch.no_grad():
        for _ in range(10):
            outputs = model(**test_input)
    inference_time = time.time() - inference_start
    
    logger.info(f"  Inference time (10 forward passes): {inference_time:.3f}s")
    logger.info(f"  Average per forward pass: {inference_time/10:.3f}s")
    
    # Log inference metrics to wandb
    if not args.no_wandb and accelerator.is_main_process:
        wandb.log({
            "inference/total_time_10_passes": inference_time,
            "inference/avg_time_per_pass": inference_time/10,
        })
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(os.path.join(args.output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
        logger.info(f"💾 Model saved to {args.output_dir}/final_model")
        
        # Evaluate perplexity at different sequence lengths (optional and OOM-safe)
        if not args.eval_skip and args.eval_max_length_factor != 0:
            logger.info("🔍 Starting perplexity evaluation for length generalization...")
            try:
                eval_max_length = int(args.max_seq_length * args.eval_max_length_factor) if args.eval_max_length_factor else None
                perplexity_results = evaluate_model_perplexity(
                    model=unwrapped_model,
                    tokenizer=tokenizer,
                    device=accelerator.device,
                    training_window=args.max_seq_length,
                    max_seqs=args.eval_max_seqs,
                    batch_size=args.eval_batch_size,
                    window_batch_size=args.eval_window_batch_size,
                    dataset_name=args.eval_dataset_name,
                    dataset_config=args.eval_dataset_config,
                    max_length=eval_max_length,
                    chunk_size=args.eval_chunk_size,
                    chunk_stride=args.eval_chunk_stride,
                    output_dir=args.output_dir,
                    step=step + 1,
                    log_wandb=not args.no_wandb,
                    use_packing=args.eval_use_packing,
                    align_prefix_pre_w=args.eval_align_prefix,
                    x_axis=args.eval_x_axis,
                )
                logger.info(f"✅ Perplexity evaluation completed!")
                
                # Log key metrics if available
                if isinstance(perplexity_results, dict):
                    logger.info(f"📊 Eval summary: T={perplexity_results.get('max_length')}, "
                                f"W={perplexity_results.get('training_window')}, "
                                f"K={perplexity_results.get('chunk_size')}, "
                                f"S={perplexity_results.get('chunk_stride')}")
            except Exception as e:
                logger.error(f"❌ Perplexity evaluation failed: {e}")
    
    # Finish wandb run
    if not args.no_wandb and accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()