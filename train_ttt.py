#!/usr/bin/env python3
"""
Main training script for TTT models using official training recipes.
Supports all model sizes: 125m, 350m, 760m, 1b

Based on the official JAX training configurations adapted for PyTorch.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train TTT model on C4 dataset")
    
    # Model configuration
    parser.add_argument("--model_size", type=str, default="125m", 
                       choices=["125m", "350m", "760m", "1b"],
                       help="Model size configuration")
    parser.add_argument("--ttt_layer_type", type=str, default="linear",
                       choices=["linear", "mlp"],
                       help="TTT layer type")
    parser.add_argument("--ttt_base_lr", type=float, default=1.0,
                       help="TTT base learning rate")
    
    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="allenai/c4")
    parser.add_argument("--dataset_config", type=str, default="en")
    parser.add_argument("--seq_length", type=int, default=2048)
    
    # Training configuration
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=4800)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--lr_end", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=480)
    parser.add_argument("--lr_decay_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                       choices=["no", "fp16", "bf16"])
    
    # TTT-specific configuration
    parser.add_argument("--state_passing", action="store_true", default=False)
    parser.add_argument("--state_reset_interval", type=int, default=100)
    
    # Checkpointing
    parser.add_argument("--save_checkpoint_freq", type=int, default=1000)
    parser.add_argument("--save_milestone_freq", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="./experiments/ttt_train")
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    return parser.parse_args()


def tokenize_function(examples, tokenizer, seq_length):
    """Tokenize and chunk text into fixed-length sequences."""
    # Tokenize all texts
    tokenized = tokenizer(examples["text"], truncation=False, padding=False)
    
    # Concatenate all texts
    concatenated = {k: sum(tokenized[k], []) for k in tokenized.keys()}
    total_length = len(concatenated["input_ids"])
    
    # Drop the small remainder
    if total_length >= seq_length:
        total_length = (total_length // seq_length) * seq_length
    
    # Split by chunks of seq_length
    result = {
        k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
        for k, t in concatenated.items()
    }
    
    return result


def create_dataloader(args, tokenizer, accelerator):
    """Create training dataloader."""
    # Load dataset
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split="train",
        streaming=True  # Use streaming for large datasets like C4
    )
    
    # Tokenize and chunk
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.seq_length),
        batched=True,
        remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else []
    )
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=lambda batch: {
            "input_ids": torch.tensor([ex["input_ids"] for ex in batch])
        }
    )
    
    return dataloader


def save_checkpoint(model, tokenizer, optimizer, lr_scheduler, step, args, accelerator):
    """Save model checkpoint."""
    output_dir = Path(args.output_dir) / f"checkpoint-{step}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save optimizer and scheduler
    torch.save({
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "step": step,
        "args": vars(args)
    }, output_dir / "training_state.pt")
    
    accelerator.print(f"Checkpoint saved to {output_dir}")


def main():
    args = parse_args()
    
    # Set decay steps to max_train_steps if not specified
    if args.lr_decay_steps is None:
        args.lr_decay_steps = args.max_train_steps
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.wandb_project and WANDB_AVAILABLE else None,
    )
    
    # Initialize W&B
    if accelerator.is_main_process and args.wandb_project and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save args
    if accelerator.is_main_process:
        with open(Path(args.output_dir) / "training_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    config = TTTConfig(**TTT_STANDARD_CONFIGS[args.model_size])
    config.ttt_layer_type = args.ttt_layer_type
    config.ttt_base_lr = args.ttt_base_lr
    config.max_seq_length = args.seq_length
    
    model = TTTForCausalLM(config)
    
    accelerator.print(f"Model size: {args.model_size}")
    accelerator.print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create dataloader
    train_dataloader = create_dataloader(args, tokenizer, accelerator)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create learning rate scheduler (cosine decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.lr_decay_steps,
        num_cycles=0.5,  # Cosine decay to end_lr
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Training loop
    accelerator.print("***** Running training *****")
    accelerator.print(f"  Max steps = {args.max_train_steps}")
    accelerator.print(f"  Per device batch size = {args.per_device_train_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total batch size = {args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    total_loss = 0.0
    
    model.train()
    train_iterator = iter(train_dataloader)
    
    while completed_steps < args.max_train_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)
        
        with accelerator.accumulate(model):
            input_ids = batch["input_ids"]
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        if accelerator.sync_gradients:
            completed_steps += 1
            progress_bar.update(1)
            total_loss += loss.detach().float()
            
            # Logging
            if completed_steps % args.logging_steps == 0:
                avg_loss = total_loss / args.logging_steps
                current_lr = lr_scheduler.get_last_lr()[0]
                
                accelerator.print(
                    f"Step {completed_steps}: loss={avg_loss:.4f}, lr={current_lr:.2e}"
                )
                
                if accelerator.is_main_process and args.wandb_project and WANDB_AVAILABLE:
                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": current_lr,
                        "step": completed_steps
                    })
                
                total_loss = 0.0
            
            # Save checkpoint
            if completed_steps % args.save_checkpoint_freq == 0:
                save_checkpoint(model, tokenizer, optimizer, lr_scheduler, completed_steps, args, accelerator)
            
            # Save milestone
            if args.save_milestone_freq and completed_steps % args.save_milestone_freq == 0:
                output_dir = Path(args.output_dir) / f"milestone-{completed_steps}"
                output_dir.mkdir(parents=True, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                accelerator.print(f"Milestone saved to {output_dir}")
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir) / "final_model"
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        accelerator.print(f"Training complete! Final model saved to {output_dir}")
    
    if accelerator.is_main_process and args.wandb_project and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
