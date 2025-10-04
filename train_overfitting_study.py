"""
Train TTT model for overfitting study - NO early stopping.
Runs full 100k steps to observe convergence and potential overfitting.
Can be run for both W=64 and W=128 to compare overfitting characteristics.

Usage:
    python train_overfitting_study.py --max_seq_length 64 --output_dir overfitting_w64
    python train_overfitting_study.py --max_seq_length 128 --output_dir overfitting_w128
"""

import argparse
import json
import logging
import os
from pathlib import Path
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from ttt import TTTConfig, TTTForCausalLM
from perplexity_evaluator import PerplexityEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TTT model without early stopping for overfitting study")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Training window size (W)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--max_steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--num_eval_checkpoints", type=int, default=50, help="Number of evaluation checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eval_max_length", type=int, default=None, help="Max length for evaluation (default: 20×W)")
    return parser.parse_args()


def get_model_config(args):
    """Create TTTConfig with state passing enabled."""
    config = TTTConfig(
        vocab_size=50257,  # GPT-2 vocab size
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=args.max_seq_length,
        state_passing=True,  # Enable state passing
        mini_batch_size=16,
        use_cache=False,
    )
    return config


def create_data_collator(tokenizer):
    """Create data collator for language modeling."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None,  # We already pad during tokenization
    )


def run_length_gen_eval(
    model,
    tokenizer,
    device,
    args,
    checkpoint_idx,
    step,
    output_dir,
):
    """Run length generalization evaluation."""
    eval_max_length = args.eval_max_length or (args.max_seq_length * 20)
    
    evaluator = PerplexityEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        training_window=args.max_seq_length,
        max_seqs=64,
        batch_size=8,
        chunk_size=args.max_seq_length,
        chunk_stride=args.max_seq_length // 4,
        pack_stride=eval_max_length,  # Disjoint sequences
    )
    
    # Run chunk-based evaluation
    results = evaluator.evaluate_chunks_only(
        max_length=eval_max_length,
        use_packing=True,
    )
    
    # Save visualization
    ckpt_dir = output_dir / f"checkpoint_{checkpoint_idx:03d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = ckpt_dir / f"chunk_eval_W{args.max_seq_length}_T{eval_max_length}_K{args.max_seq_length}_S{args.max_seq_length//4}_ckpt{checkpoint_idx}.png"
    evaluator.plot_chunks(
        res=results,
        save_path=str(plot_path),
        show=False,
    )
    
    # Extract key metrics at specific positions
    chunk_centers = results["chunk_centers"]
    ttt_ppls = results["chunk_ppl_ttt"]
    sw_ppls = results["chunk_ppl_sw"]
    
    # Find PPL at 1×W, 2×W, 5×W, 10×W, 20×W
    def find_nearest_ppl(position):
        """Find PPL closest to target position."""
        idx = min(range(len(chunk_centers)), key=lambda i: abs(chunk_centers[i] - position))
        return ttt_ppls[idx], sw_ppls[idx], chunk_centers[idx]
    
    metrics = {
        "checkpoint_idx": checkpoint_idx,
        "step": step,
        "training_window": args.max_seq_length,
        "eval_max_length": eval_max_length,
        "chunk_size": args.max_seq_length,
        "chunk_stride": args.max_seq_length // 4,
    }
    
    # Add PPL at key positions
    for multiplier in [1, 2, 5, 10, 20]:
        target_pos = args.max_seq_length * multiplier
        if target_pos <= eval_max_length:
            ttt_ppl, sw_ppl, actual_pos = find_nearest_ppl(target_pos)
            metrics[f"ttt_ppl_at_{multiplier}xW"] = ttt_ppl
            metrics[f"sw_ppl_at_{multiplier}xW"] = sw_ppl
            metrics[f"x_pos_at_{multiplier}xW"] = actual_pos
    
    # Add overall statistics
    metrics["ttt_ppl_mean"] = float(sum(ttt_ppls) / len(ttt_ppls))
    metrics["ttt_ppl_min"] = float(min(ttt_ppls))
    metrics["ttt_ppl_max"] = float(max(ttt_ppls))
    metrics["sw_ppl_mean"] = float(sum(sw_ppls) / len(sw_ppls))
    metrics["sw_ppl_min"] = float(min(sw_ppls))
    metrics["sw_ppl_max"] = float(max(sw_ppls))
    
    # Save summary
    summary_path = ckpt_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("TTT OVERFITTING STUDY - NO EARLY STOPPING")
    logger.info("=" * 80)
    logger.info(f"Training window (W): {args.max_seq_length}")
    logger.info(f"Max training steps: {args.max_steps:,}")
    logger.info(f"Evaluation checkpoints: {args.num_eval_checkpoints}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Evaluation max length: {args.eval_max_length or args.max_seq_length * 20} ({(args.eval_max_length or args.max_seq_length * 20) // args.max_seq_length}×W)")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    logger.info("\n🏗️  Initializing model...")
    config = get_model_config(args)
    model = TTTForCausalLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize tokenizer
    logger.info("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info("\n📚 Loading WikiText-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    
    # Filter out very short sequences
    logger.info("  Filtering short sequences...")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)
    logger.info(f"  Training examples after filtering: {len(dataset):,}")
    
    # Tokenization function with fixed-length padding
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
        )
        # Create labels (copy of input_ids for causal LM)
        tokenized["labels"] = [[token_id for token_id in ids] for ids in tokenized["input_ids"]]
        return tokenized
    
    logger.info("  Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    # Create data collator
    data_collator = create_data_collator(tokenizer)
    
    # Calculate evaluation schedule
    eval_steps = [0]  # Always evaluate at step 0 (untrained)
    if args.num_eval_checkpoints > 1:
        step_interval = args.max_steps // (args.num_eval_checkpoints - 1)
        eval_steps.extend([step_interval * i for i in range(1, args.num_eval_checkpoints)])
    
    logger.info(f"\n📊 Evaluation schedule: {args.num_eval_checkpoints} checkpoints")
    logger.info(f"   First 5: {eval_steps[:5]}")
    logger.info(f"   Last 5: {eval_steps[-5:]}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "training_checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        warmup_steps=int(args.max_steps * 0.02),  # 2% warmup
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=50,
        save_strategy="no",  # We'll save manually at eval checkpoints
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
    )
    
    logger.info(f"\n🎯 Training configuration:")
    logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Warmup steps: {training_args.warmup_steps:,}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Weight decay: {training_args.weight_decay}")
    logger.info(f"  LR scheduler: {training_args.lr_scheduler_type}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Run initial evaluation (untrained model)
    logger.info("\n📊 Running initial evaluation (untrained model)...")
    logger.info(f"📊 Checkpoint 0/{args.num_eval_checkpoints} (step 0): Running length gen eval...")
    
    metrics_0 = run_length_gen_eval(
        model=model,
        tokenizer=tokenizer,
        device=device,
        args=args,
        checkpoint_idx=0,
        step=0,
        output_dir=output_dir,
    )
    
    logger.info(f"✅ Checkpoint 0: Eval complete. Summary saved to {output_dir}/checkpoint_000/summary.json")
    logger.info(f"   @1xW: TTT={metrics_0['ttt_ppl_at_1xW']:.1f}, SW={metrics_0['sw_ppl_at_1xW']:.1f}")
    if 'ttt_ppl_at_10xW' in metrics_0:
        logger.info(f"   @10xW: TTT={metrics_0['ttt_ppl_at_10xW']:.1f}, SW={metrics_0['sw_ppl_at_10xW']:.1f}")
    
    # Training loop with periodic evaluation
    logger.info(f"\n🚀 Starting training for {args.max_steps:,} steps (NO EARLY STOPPING)...")
    start_time = time.time()
    
    current_eval_idx = 1
    next_eval_step = eval_steps[current_eval_idx] if current_eval_idx < len(eval_steps) else None
    
    class EvalCallback:
        """Callback to perform evaluation at specific steps."""
        def __init__(self, eval_steps, output_dir, args, model, tokenizer, device):
            self.eval_steps = eval_steps[1:]  # Skip step 0 (already done)
            self.output_dir = output_dir
            self.args = args
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.current_idx = 1
            
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in self.eval_steps:
                logger.info(f"\n📊 Checkpoint {self.current_idx}/{self.args.num_eval_checkpoints} (step {state.global_step}): Running length gen eval...")
                
                metrics = run_length_gen_eval(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    args=self.args,
                    checkpoint_idx=self.current_idx,
                    step=state.global_step,
                    output_dir=self.output_dir,
                )
                
                logger.info(f"✅ Checkpoint {self.current_idx}: Eval complete. Summary saved to {self.output_dir}/checkpoint_{self.current_idx:03d}/summary.json")
                logger.info(f"   @1xW: TTT={metrics['ttt_ppl_at_1xW']:.1f}, SW={metrics['sw_ppl_at_1xW']:.1f}")
                if 'ttt_ppl_at_10xW' in metrics:
                    logger.info(f"   @10xW: TTT={metrics['ttt_ppl_at_10xW']:.1f}, SW={metrics['sw_ppl_at_10xW']:.1f}")
                
                self.current_idx += 1
            
            return control
    
    from transformers import TrainerCallback
    
    class CustomEvalCallback(TrainerCallback):
        def __init__(self, eval_steps, output_dir, args, model, tokenizer, device):
            self.eval_steps = set(eval_steps[1:])  # Skip step 0
            self.output_dir = output_dir
            self.args = args
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.current_idx = 1
            
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in self.eval_steps:
                logger.info(f"\n📊 Checkpoint {self.current_idx}/{self.args.num_eval_checkpoints} (step {state.global_step}): Running length gen eval...")
                
                metrics = run_length_gen_eval(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    args=self.args,
                    checkpoint_idx=self.current_idx,
                    step=state.global_step,
                    output_dir=self.output_dir,
                )
                
                logger.info(f"✅ Checkpoint {self.current_idx}: Eval complete. Summary saved to {self.output_dir}/checkpoint_{self.current_idx:03d}/summary.json")
                logger.info(f"   @1xW: TTT={metrics['ttt_ppl_at_1xW']:.1f}, SW={metrics['sw_ppl_at_1xW']:.1f}")
                if 'ttt_ppl_at_10xW' in metrics:
                    logger.info(f"   @10xW: TTT={metrics['ttt_ppl_at_10xW']:.1f}, SW={metrics['sw_ppl_at_10xW']:.1f}")
                
                self.current_idx += 1
                self.eval_steps.remove(state.global_step)
    
    # Add callback
    eval_callback = CustomEvalCallback(
        eval_steps=eval_steps,
        output_dir=output_dir,
        args=args,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    trainer.add_callback(eval_callback)
    
    # Train
    trainer.train()
    
    training_time = (time.time() - start_time) / 60
    
    # Save final model
    final_model_dir = output_dir / "final_model"
    logger.info(f"\n💾 Saving final model to {final_model_dir}")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Total time: {training_time:.1f}m")
    logger.info(f"  Total steps: {args.max_steps:,}")
    logger.info(f"  Eval checkpoints: {args.num_eval_checkpoints}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
