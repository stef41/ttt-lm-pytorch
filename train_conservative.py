#!/usr/bin/env python3
"""
Very conservative TTT training with careful initialization and tiny learning rates.
This approach focuses on training stability over speed.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
import argparse
import logging
import os
from accelerate import Accelerator
import time
import math
import wandb
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS
from perplexity_evaluator import evaluate_model_perplexity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_model_carefully(config):
    """Initialize TTT model with very conservative weight initialization."""
    model = TTTForCausalLM(config)
    
    # Apply careful initialization
    for name, param in model.named_parameters():
        if param.dim() > 1:
            if 'embed' in name:
                # Small embedding initialization
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif any(x in name.lower() for x in ['w1', 'w2', 'ttt']):
                # Very small TTT layer initialization
                nn.init.normal_(param, mean=0.0, std=0.001)
            elif 'lm_head' in name:
                # Small head initialization
                nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                # Standard initialization for other layers
                nn.init.normal_(param, mean=0.0, std=0.02)
        else:
            # Bias terms
            nn.init.zeros_(param)
    
    logger.info("✅ Applied conservative weight initialization")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="125m")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)  # Very small batch
    parser.add_argument("--learning_rate", type=float, default=1e-5)  # Very low LR
    parser.add_argument("--max_train_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=20)  # Longer warmup
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--max_seq_length", type=int, default=64)  # Set to 64 tokens for all experiments
    parser.add_argument("--output_dir", default="./conservative_ttt_model")
    parser.add_argument("--cache_reset_interval", type=int, default=20)  # Frequent resets
    parser.add_argument("--grad_clip", type=float, default=0.5)  # Aggressive grad clipping
    parser.add_argument("--state_passing", action="store_true", help="Enable state passing between sequences")
    parser.add_argument("--wandb_project", type=str, default="ttt-conservative-training", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb_offline", action="store_true", help="Force W&B offline mode")
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    
    # Initialize wandb
    if not args.no_wandb and accelerator.is_main_process:
        wandb_mode = "offline" if args.wandb_offline else "online"
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"conservative-{args.model_size}-lr{args.learning_rate}-bs{args.per_device_train_batch_size}",
            mode=wandb_mode,  # Online by default, offline if specified
            config={
                "model_size": args.model_size,
                "learning_rate": args.learning_rate,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "max_train_steps": args.max_train_steps,
                "warmup_steps": args.warmup_steps,
                "max_seq_length": args.max_seq_length,
                "grad_clip": args.grad_clip,
                "state_passing": args.state_passing,
                "cache_reset_interval": args.cache_reset_interval,
                "training_type": "conservative",
                "mixed_precision": "bf16"
            }
        )
    
    logger.info(f"🚀 Conservative TTT Training:")
    logger.info(f"  Device: {accelerator.device}")
    logger.info(f"  Model size: {args.model_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Batch size: {args.per_device_train_batch_size}")
    logger.info(f"  Grad clip: {args.grad_clip}")
    logger.info(f"  State passing: {args.state_passing}")
    logger.info(f"  W&B logging: {'enabled' if not args.no_wandb else 'disabled'}")
    if not args.no_wandb:
        logger.info(f"  W&B mode: {'offline' if args.wandb_offline else 'online'}")
    
    # Load config and initialize model
    config = TTT_STANDARD_CONFIGS[args.model_size].copy()
    config["state_passing"] = args.state_passing  # Use command line argument
    config["max_position_embeddings"] = args.max_seq_length
    config = TTTConfig(**config)
    
    model = initialize_model_carefully(config)
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Log model info to wandb
    if not args.no_wandb and accelerator.is_main_process:
        wandb.log({
            "model/total_parameters": sum(p.numel() for p in model.parameters()),
            "model/trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model/hidden_size": config.hidden_size,
            "model/num_layers": config.num_hidden_layers,
            "model/num_heads": config.num_attention_heads,
        })
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info("📁 Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")  # Full dataset for proper training
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
            return_tensors="pt"
        )
    
    # Tokenize and filter
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["text"].strip()) > 10)
    
    logger.info(f"  Training examples: {len(tokenized_dataset)}")
    
    # Create dataloader
    def collate_fn(examples):
        input_ids = torch.stack([torch.tensor(ex["input_ids"]) for ex in examples])
        attention_mask = torch.stack([torch.tensor(ex["attention_mask"]) for ex in examples])
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Setup optimizer with very conservative settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),  # More conservative beta2
        eps=1e-8
    )
    
    # Setup scheduler with long warmup
    total_steps = min(len(dataloader), args.max_train_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Prepare with accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    logger.info("🏃 Starting conservative training...")
    
    model.train()
    step = 0
    start_time = time.time()
    loss_history = []
    
    for batch in dataloader:
        if step >= args.max_train_steps:
            break
            
        # Reset cache frequently for stability (only if cache_reset_interval > 0)
        if args.cache_reset_interval > 0 and step % args.cache_reset_interval == 0 and step > 0:
            if hasattr(model.module if hasattr(model, 'module') else model, 'reset_cache'):
                (model.module if hasattr(model, 'module') else model).reset_cache()
                logger.info(f"🔄 Cache reset at step {step}")
        elif args.cache_reset_interval == 0 and step == 0:
            logger.info("🔗 Persistent state passing enabled (cache never resets)")
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Check for loss explosion
        if torch.isnan(loss) or loss.item() > 50.0:
            logger.warning(f"⚠️  Loss explosion detected: {loss.item():.4f}")
            
            # Log warning to wandb
            if not args.no_wandb and accelerator.is_main_process:
                wandb.log({
                    "warnings/loss_explosion": loss.item(),
                    "warnings/step": step,
                }, step=step)
                
            # Skip this batch
            optimizer.zero_grad()
            step += 1
            continue
            
        # Backward pass
        accelerator.backward(loss)
        
        # Very aggressive gradient clipping
        accelerator.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        
        # Check gradient norms
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        if total_grad_norm > 10.0:
            logger.warning(f"⚠️  Large gradient norm: {total_grad_norm:.4f}")
            
            # Log warning to wandb
            if not args.no_wandb and accelerator.is_main_process:
                wandb.log({
                    "warnings/large_grad_norm": total_grad_norm,
                    "warnings/step": step,
                }, step=step)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Track loss history
        loss_history.append(loss.item())
        
        # Logging
        if step % args.logging_steps == 0:
            lr = scheduler.get_last_lr()[0]
            avg_loss = sum(loss_history[-10:]) / min(len(loss_history), 10)
            logger.info(f"Step {step:3d}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, lr={lr:.2e}, grad_norm={total_grad_norm:.4f}")
            
            # Wandb logging
            if not args.no_wandb and accelerator.is_main_process:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/learning_rate": lr,
                    "train/grad_norm": total_grad_norm,
                    "train/step": step,
                    "train/epoch": step / len(dataloader),
                }, step=step)
        
        step += 1
    
    # Save model
    logger.info("💾 Saving model...")
    output_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    total_time = time.time() - start_time
    logger.info(f"✅ Training complete! Time: {total_time:.1f}s")
    logger.info(f"📁 Model saved to {output_dir}")
    
    # Evaluate perplexity at different sequence lengths
    if accelerator.is_main_process:
        logger.info("🔍 Starting perplexity evaluation for length generalization...")
        try:
            perplexity_results = evaluate_model_perplexity(
                model=unwrapped_model,
                tokenizer=tokenizer,
                device=accelerator.device,
                training_window=args.max_seq_length,
                output_dir=args.output_dir,
                step=step,
                log_wandb=not args.no_wandb
            )
            logger.info(f"✅ Perplexity evaluation completed!")
            
            # Log key metrics
            min_ppl = min(perplexity_results['perplexities'])
            max_ppl = max(perplexity_results['perplexities'])
            logger.info(f"📊 Perplexity range: {min_ppl:.3f} - {max_ppl:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Perplexity evaluation failed: {e}")
    
    # Loss analysis
    if len(loss_history) > 10:
        initial_loss = sum(loss_history[:5]) / 5
        final_loss = sum(loss_history[-5:]) / 5
        loss_reduction = initial_loss - final_loss
        logger.info(f"📊 Loss analysis:")
        logger.info(f"  Initial average loss: {initial_loss:.4f}")
        logger.info(f"  Final average loss: {final_loss:.4f}")
        logger.info(f"  Loss reduction: {loss_reduction:.4f}")
        
        # Final wandb logging
        if not args.no_wandb and accelerator.is_main_process:
            wandb.log({
                "summary/initial_loss": initial_loss,
                "summary/final_loss": final_loss,
                "summary/loss_reduction": loss_reduction,
                "summary/training_time": total_time,
                "summary/steps_completed": step,
                "summary/samples_processed": step * args.per_device_train_batch_size,
            })
            
            # Log loss history as a plot
            loss_steps = list(range(len(loss_history)))
            wandb.log({
                "charts/loss_history": wandb.plot.line_series(
                    xs=loss_steps,
                    ys=[loss_history],
                    keys=["loss"],
                    title="Training Loss Over Time",
                    xname="Step"
                )
            })
            
            wandb.finish()

if __name__ == "__main__":
    main()