#!/usr/bin/env python3
"""
Extended curriculum training for full convergence study.

Based on previous run (15k steps, still improving), we extend to 100k+ steps
to reach true convergence and study length generalization emergence properly.

Key changes from short run:
- 100k total steps (vs 15k)
- 50 evaluation checkpoints (vs 30) for finer granularity
- Early stopping with patience to avoid overtraining
- Save best model based on eval perplexity
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

from ttt import TTTConfig, TTTForCausalLM
from perplexity_evaluator import PerplexityEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    
    # Model
    p.add_argument("--model_size", type=str, default="125m", choices=["125m"])
    p.add_argument("--max_seq_length", type=int, default=64, help="Training window W")
    
    # Training
    p.add_argument("--max_train_steps", type=int, default=100000, help="Total training steps")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)
    
    # Dataset
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--dataset_subset_size", type=int, default=-1, help="-1 for full dataset")
    
    # TTT specific
    p.add_argument("--state_passing", action="store_true", help="Enable state passing")
    p.add_argument("--state_reset_interval", type=int, default=0, help="0 = never reset")
    
    # Curriculum evaluation
    p.add_argument("--num_eval_checkpoints", type=int, default=50, help="Number of evaluation checkpoints")
    p.add_argument("--eval_max_length", type=int, default=1280, help="Max eval length T (should be ~20×W)")
    p.add_argument("--eval_batch_size", type=int, default=4)
    
    # Early stopping
    p.add_argument("--early_stop_patience", type=int, default=10, help="Stop if no improvement for N checkpoints")
    p.add_argument("--early_stop_min_delta", type=float, default=0.01, help="Minimum improvement to count")
    
    # Output
    p.add_argument("--output_dir", type=str, default="curriculum_exp_long")
    p.add_argument("--save_best_only", action="store_true", help="Only save best model")
    
    return p.parse_args()


def get_model_config(args):
    """Get model configuration."""
    if args.model_size == "125m":
        config = TTTConfig(
            vocab_size=50257,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=2048,
            mini_batch_size=args.max_seq_length,
            state_passing=args.state_passing,  # Fixed: was enable_state_passing
        )
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    return config


def create_data_collator(tokenizer):
    """Create data collator for language modeling."""
    from transformers import DataCollatorForLanguageModeling
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None,  # Don't pad since we already pad to max_length
    )


def run_length_gen_eval(args, model, tokenizer, accelerator, checkpoint_idx, step, output_dir):
    """
    Run length generalization evaluation at a checkpoint.
    
    Returns:
        dict: Summary statistics including PPL at key positions
    """
    try:
        eval_dir = output_dir / f"checkpoint_{checkpoint_idx:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 Checkpoint {checkpoint_idx}/{args.num_eval_checkpoints} (step {step}): Running length gen eval...")
        
        # Create evaluator
        evaluator = PerplexityEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=accelerator.device,
            training_window=args.max_seq_length,
            batch_size=args.eval_batch_size,
            max_seqs=64,
        )
        
        # Run evaluation using evaluate_chunks_only
        results = evaluator.evaluate_chunks_only(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            max_length=args.eval_max_length,
            use_packing=True,
        )
        
        # Save plot
        plot_path = eval_dir / f"chunk_eval_W{args.max_seq_length}_T{args.eval_max_length}_K{results['chunk_size']}_S{results['chunk_stride']}_ckpt{checkpoint_idx}.png"
        evaluator.plot_chunks(
            results,
            save_path=str(plot_path),
            show=False,
        )
        
        # Extract summary statistics at key positions
        def find_closest_idx(positions, target):
            return int(np.argmin(np.abs(np.array(positions) - target)))
        
        positions = results['chunk_centers']
        ttt_ppls = results['chunk_ppl_ttt']
        sw_ppls = results['chunk_ppl_sw']
        W = args.max_seq_length
        
        summary = {
            'checkpoint_idx': checkpoint_idx,
            'step': step,
            'training_window': W,
            'eval_max_length': args.eval_max_length,
            'chunk_size': results['chunk_size'],
            'chunk_stride': results['chunk_stride'],
        }
        
        # Sample at 1×W, 2×W, 5×W, 10×W, 20×W
        for mult in [1, 2, 5, 10, 20]:
            target_pos = W * mult
            if target_pos <= args.eval_max_length:
                idx = find_closest_idx(positions, target_pos)
                summary[f'ttt_ppl_at_{mult}xW'] = float(ttt_ppls[idx])
                summary[f'sw_ppl_at_{mult}xW'] = float(sw_ppls[idx])
                summary[f'x_pos_at_{mult}xW'] = int(positions[idx])
        
        # Overall statistics
        summary['ttt_ppl_mean'] = float(np.mean(ttt_ppls))
        summary['ttt_ppl_min'] = float(np.min(ttt_ppls))
        summary['ttt_ppl_max'] = float(np.max(ttt_ppls))
        summary['sw_ppl_mean'] = float(np.mean(sw_ppls))
        summary['sw_ppl_min'] = float(np.min(sw_ppls))
        summary['sw_ppl_max'] = float(np.max(sw_ppls))
        
        # Save summary
        summary_path = eval_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Checkpoint {checkpoint_idx}: Eval complete. Summary saved to {summary_path}")
        if 'ttt_ppl_at_1xW' in summary:
            logger.info(f"   @1xW: TTT={summary['ttt_ppl_at_1xW']:.1f}, SW={summary['sw_ppl_at_1xW']:.1f}")
        if 'ttt_ppl_at_10xW' in summary:
            logger.info(f"   @10xW: TTT={summary['ttt_ppl_at_10xW']:.1f}, SW={summary['sw_ppl_at_10xW']:.1f}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Checkpoint {checkpoint_idx}: Eval failed: {e}")
        import traceback
        traceback.print_exc()
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
    
    logger.info("🚀 Extended Curriculum Training for Length Generalization")
    logger.info(f"  Model: {args.model_size}, W={args.max_seq_length}, eval T={args.eval_max_length}")
    logger.info(f"  Total steps: {args.max_train_steps}, Eval checkpoints: {args.num_eval_checkpoints}")
    logger.info(f"  Early stopping: patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}")
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
        # Tokenize with padding and truncation to ensure fixed length
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
        )
        # Create labels (same as input_ids for causal LM)
        tokenized["labels"] = [[token_id for token_id in ids] for ids in tokenized["input_ids"]]
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
        collate_fn=data_collator,
        shuffle=True,
    )
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        foreach=True,
    )
    
    warmup_steps = int(args.max_train_steps * args.warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        'model_size': args.model_size,
        'training_window': args.max_seq_length,
        'eval_max_length': args.eval_max_length,
        'max_train_steps': args.max_train_steps,
        'num_eval_checkpoints': args.num_eval_checkpoints,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_ratio': args.warmup_ratio,
        'early_stop_patience': args.early_stop_patience,
        'seed': args.seed,
        'start_time': datetime.now().isoformat(),
    }
    
    # Compute evaluation steps (equally spaced)
    eval_steps = [int(i * args.max_train_steps / (args.num_eval_checkpoints - 1)) 
                  for i in range(args.num_eval_checkpoints)]
    eval_steps[0] = 0  # Always eval at step 0
    metadata['eval_steps'] = eval_steps
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"📊 Evaluation schedule: {len(eval_steps)} checkpoints")
    logger.info(f"   First 5: {eval_steps[:5]}")
    logger.info(f"   Last 5: {eval_steps[-5:]}")
    
    # Training loop
    global_step = 0
    checkpoint_idx = 0
    all_summaries = []
    best_ppl_at_1xW = float('inf')
    best_checkpoint_idx = -1
    patience_counter = 0
    
    start_time = datetime.now()
    
    # Initial evaluation (step 0)
    logger.info("📊 Running initial evaluation (untrained model)...")
    summary = run_length_gen_eval(args, model, tokenizer, accelerator, checkpoint_idx, 0, output_dir)
    if summary:
        all_summaries.append(summary)
        if 'ttt_ppl_at_1xW' in summary:
            best_ppl_at_1xW = summary['ttt_ppl_at_1xW']
    checkpoint_idx += 1
    
    # Training
    model.train()
    progress_bar = tqdm(total=args.max_train_steps, desc="Training", disable=not accelerator.is_local_main_process)
    
    epoch = 0
    while global_step < args.max_train_steps:
        epoch += 1
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                
                # Log every 50 steps
                if global_step % 50 == 0:
                    train_ppl = torch.exp(loss).item()
                    lr = lr_scheduler.get_last_lr()[0]
                    logger.info(f"Step {global_step:5d}/{args.max_train_steps}: loss={loss.item():.4f} (ppl~{train_ppl:.1f}), lr={lr:.2e}")
                
                # Checkpoint evaluation
                if global_step in eval_steps and checkpoint_idx < args.num_eval_checkpoints:
                    logger.info(f"📊 Checkpoint {checkpoint_idx}/{args.num_eval_checkpoints} (step {global_step}): Running length gen eval...")
                    
                    summary = run_length_gen_eval(args, model, tokenizer, accelerator, checkpoint_idx, global_step, output_dir)
                    
                    if summary:
                        all_summaries.append(summary)
                        
                        # Check for improvement (early stopping)
                        if 'ttt_ppl_at_1xW' in summary:
                            current_ppl = summary['ttt_ppl_at_1xW']
                            improvement = best_ppl_at_1xW - current_ppl
                            
                            if improvement > args.early_stop_min_delta:
                                logger.info(f"✅ New best PPL@1xW: {current_ppl:.1f} (improved by {improvement:.1f})")
                                best_ppl_at_1xW = current_ppl
                                best_checkpoint_idx = checkpoint_idx
                                patience_counter = 0
                                
                                # Save best model
                                if accelerator.is_main_process:
                                    best_model_dir = output_dir / "best_model"
                                    best_model_dir.mkdir(parents=True, exist_ok=True)
                                    unwrapped_model = accelerator.unwrap_model(model)
                                    unwrapped_model.save_pretrained(best_model_dir)
                                    tokenizer.save_pretrained(best_model_dir)
                                    logger.info(f"💾 Best model saved to {best_model_dir}")
                            else:
                                patience_counter += 1
                                logger.info(f"⚠️  No improvement ({improvement:.1f}). Patience: {patience_counter}/{args.early_stop_patience}")
                                
                                if patience_counter >= args.early_stop_patience:
                                    logger.info(f"🛑 Early stopping triggered! No improvement for {args.early_stop_patience} checkpoints.")
                                    logger.info(f"   Best checkpoint: {best_checkpoint_idx} with PPL@1xW={best_ppl_at_1xW:.1f}")
                                    break
                    
                    checkpoint_idx += 1
                    model.train()
                
                # Save summaries periodically
                if global_step % 1000 == 0 and accelerator.is_main_process:
                    with open(output_dir / "all_summaries.json", "w") as f:
                        json.dump(all_summaries, f, indent=2)
            
            if global_step >= args.max_train_steps:
                break
        
        # Check early stopping between epochs too
        if patience_counter >= args.early_stop_patience:
            break
    
    # Final model save
    if accelerator.is_main_process:
        final_model_dir = output_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        logger.info(f"💾 Final model saved to {final_model_dir}")
    
    # Save all summaries
    if accelerator.is_main_process:
        with open(output_dir / "all_summaries.json", "w") as f:
            json.dump(all_summaries, f, indent=2)
        
        # Update metadata with results
        metadata['end_time'] = datetime.now().isoformat()
        metadata['total_steps_completed'] = global_step
        metadata['total_checkpoints_evaluated'] = len(all_summaries)
        metadata['best_checkpoint_idx'] = best_checkpoint_idx
        metadata['best_ppl_at_1xW'] = best_ppl_at_1xW
        metadata['early_stopped'] = patience_counter >= args.early_stop_patience
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    elapsed = datetime.now() - start_time
    logger.info(f"\n✅ Curriculum training complete!")
    logger.info(f"  Total time: {elapsed.total_seconds()/60:.1f}m")
    logger.info(f"  Total steps: {global_step}")
    logger.info(f"  Eval checkpoints: {len(all_summaries)}")
    logger.info(f"  Best checkpoint: {best_checkpoint_idx} (PPL@1xW={best_ppl_at_1xW:.1f})")


if __name__ == "__main__":
    main()
