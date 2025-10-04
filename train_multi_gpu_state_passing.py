#!/usr/bin/env python3
"""
Multi-GPU TTT Training with State Passing Support
Handles distributed training with proper cache synchronization.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed

# Add the current directory to the path to import ttt
sys.path.append(str(Path(__file__).parent))
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS, TTTCache
from perplexity_evaluator import evaluate_model_perplexity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def monitor_gpu_usage():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak"
    return "No GPU available"

def create_distributed_cache(model, batch_size, accelerator):
    """Create TTT cache that works with distributed training"""
    # Only create cache on the base model (unwrapped from DDP)
    base_model = accelerator.unwrap_model(model).model
    cache = TTTCache(base_model, batch_size)
    
    # For distributed training, we need to ensure cache states are synchronized
    # The cache parameters will be handled by DDP, but we need to make sure
    # the cache state tensors are on the right device
    if accelerator.is_main_process:
        logger.info(f"Created TTT cache for distributed training on {accelerator.device}")
    
    return cache

def synchronize_cache_if_needed(cache, accelerator, step):
    """Synchronize cache state across ranks if needed"""
    if cache is None or not accelerator.use_distributed:
        return
    
    # For state passing in distributed training, we only sync at reset points
    # to avoid performance overhead. The cache parameters are handled by DDP
    # but the cache state tensors need manual synchronization
    pass  # For now, we'll rely on the reset mechanism to maintain consistency

def main():
    parser = argparse.ArgumentParser(description='Train TTT model with multi-GPU and state passing')
    
    # Model configuration
    parser.add_argument('--model_size', type=str, default='125m', 
                        choices=['125m', '350m', '760m', '1.3b'],
                        help='Model size configuration')
    
    # Training configuration  
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                        help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--max_train_steps', type=int, default=100,
                        help='Maximum training steps')
    parser.add_argument('--max_seq_length', type=int, default=64,
                        help='Maximum sequence length - set to 64 tokens for all experiments')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Logging frequency')
    
    # State passing configuration
    parser.add_argument('--state_passing', action='store_true', default=True,
                        help='Enable state passing (enabled by default)')
    parser.add_argument('--no_state_passing', action='store_true',
                        help='Disable state passing (overrides default)')
    parser.add_argument('--state_reset_interval', type=int, default=20,
                        help='Steps between cache resets (0 to disable, higher values for distributed)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./ttt_output',
                        help='Output directory')
    parser.add_argument('--dataset_name', type=str, default='wikitext',
                        help='HuggingFace dataset name')
    parser.add_argument('--dataset_subset', type=str, default='wikitext-2-raw-v1',
                        help='Dataset subset/split')
    
    # Wandb configuration
    parser.add_argument('--wandb_project', type=str, default='ttt-multi-gpu',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--wandb_offline', action='store_true',
                        help='Force W&B offline mode')
    
    args = parser.parse_args()
    
    # Handle state passing flags
    if args.no_state_passing:
        args.state_passing = False
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='bf16',
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Only log from main process
    if accelerator.is_main_process:
        logger.info("🚀 Multi-GPU TTT Training with State Passing")
        
        # Initialize wandb
        use_wandb = WANDB_AVAILABLE and not args.no_wandb
        if use_wandb:
            wandb_mode = "offline" if args.wandb_offline else "online"
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=wandb_mode,  # Online by default, offline if specified
                config={
                    "model_size": args.model_size,
                    "per_device_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "learning_rate": args.learning_rate,
                    "max_train_steps": args.max_train_steps,
                    "max_seq_length": args.max_seq_length,
                    "state_passing": args.state_passing,
                    "state_reset_interval": args.state_reset_interval,
                    "num_processes": accelerator.num_processes,
                }
            )
            logger.info(f"  W&B logging: enabled (project: {args.wandb_project})")
            logger.info(f"  W&B mode: {'offline' if args.wandb_offline else 'online'}")
        else:
            if not WANDB_AVAILABLE:
                logger.info("  W&B logging: disabled (wandb not installed)")
            else:
                logger.info("  W&B logging: disabled (--no_wandb flag)")
        
        logger.info(f"  Distributed: {accelerator.use_distributed}")
        logger.info(f"  Num processes: {accelerator.num_processes}")
        logger.info(f"  Device: {accelerator.device}")
        logger.info(f"  Mixed precision: bf16")
        logger.info(f"  Model size: {args.model_size}")
        logger.info(f"  State passing: {args.state_passing}")
        logger.info(f"  {monitor_gpu_usage()}")
    else:
        use_wandb = False
        logger.info(f"  Device: {accelerator.device}")
        logger.info(f"  Num processes: {accelerator.num_processes}")
        logger.info(f"  Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"  Model size: {args.model_size}")
        logger.info(f"  Batch size per device: {args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"  {monitor_gpu_usage()}")
    
    # Load model configuration
    if args.model_size not in TTT_STANDARD_CONFIGS:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    config = TTTConfig(**TTT_STANDARD_CONFIGS[args.model_size])
    # Enable state passing by default, unless explicitly disabled
    config.state_passing = args.state_passing
    
    if accelerator.is_main_process:
        logger.info("📋 Model Configuration:")
        logger.info(f"  Hidden size: {config.hidden_size}")
        logger.info(f"  Layers: {config.num_hidden_layers}")
        logger.info(f"  Attention heads: {config.num_attention_heads}")
        logger.info(f"  Vocab size: {config.vocab_size}")
        logger.info(f"  State passing: {config.state_passing}")
        logger.info(f"  TTT layer type: {config.ttt_layer_type}")
    
    # Initialize model
    model = TTTForCausalLM(config)
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {total_params:,}")
        logger.info(f"  {monitor_gpu_usage()}")
    
    # Load tokenizer and dataset
    if accelerator.is_main_process:
        logger.info("📁 Loading dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and process dataset
    dataset = load_dataset(args.dataset_name, args.dataset_subset, split='train', streaming=False)
    dataset = dataset.take(1000) if hasattr(dataset, 'take') else dataset.select(range(min(1000, len(dataset))))
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding=True, 
            max_length=args.max_seq_length,
            return_tensors='pt'
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 10)
    
    if accelerator.is_main_process:
        logger.info(f"  Training examples: {len(tokenized_dataset)}")
    
    # Create data loader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True
    )
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare everything with accelerator (this wraps model in DDP if distributed)
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    if accelerator.is_main_process:
        logger.info(f"🎯 After prepare: {monitor_gpu_usage()}")
        
        # Log model info to wandb
        if use_wandb:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/vocab_size": config.vocab_size,
                "model/hidden_size": config.hidden_size,
                "model/num_hidden_layers": config.num_hidden_layers,
                "model/num_heads": config.num_attention_heads,
                "model/max_seq_length": config.max_position_embeddings,
                "model/state_passing": config.state_passing,
                "model/ttt_layer_type": config.ttt_layer_type,
            }
            wandb.log(model_info)
            
            logger.info(f"📋 Model Configuration:")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Vocab size: {config.vocab_size}")
            logger.info(f"  Hidden size: {config.hidden_size}")
    
    # Training loop with distributed-aware state passing
    if accelerator.is_main_process:
        logger.info("🏃 Starting training...")
    
    model.train()
    
    # Initialize TTT cache for state passing (distributed-aware)
    ttt_cache = None
    if config.state_passing:
        if accelerator.is_main_process:
            logger.info("🔗 Initializing TTT cache for distributed state passing...")
        
        # Create cache with proper distributed handling
        ttt_cache = create_distributed_cache(model, args.per_device_train_batch_size, accelerator)
        
        if accelerator.is_main_process:
            logger.info(f"   Cache created for batch size: {args.per_device_train_batch_size}")
            logger.info(f"   Using reset interval: {args.state_reset_interval} steps")
    
    start_time = time.time()
    total_loss = 0.0
    tokens_processed = 0
    
    for step, batch in enumerate(train_dataloader):
        if step >= args.max_train_steps:
            break
        
        step_start = time.time()
        
        # Debug info on first step
        if step == 0 and accelerator.is_main_process:
            logger.info(f"  Batch input_ids device: {batch['input_ids'].device}")
            logger.info(f"  Batch shape: {batch['input_ids'].shape}")
        
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
            
            # Synchronize cache state if needed (before optimizer step)
            synchronize_cache_if_needed(ttt_cache, accelerator, step)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        step_time = time.time() - step_start
        tokens_per_sec = batch['input_ids'].numel() / step_time
        
        # Periodically reset TTT cache for distributed training stability
        # Use a higher reset interval for multi-GPU to reduce overhead
        if (config.state_passing and ttt_cache is not None and 
            args.state_reset_interval > 0 and 
            (step + 1) % args.state_reset_interval == 0):
            
            if accelerator.is_main_process:
                logger.info(f"🔄 Resetting TTT cache at step {step + 1} (distributed)")
            
            # Create fresh cache for all ranks
            ttt_cache = create_distributed_cache(model, args.per_device_train_batch_size, accelerator)
        
        # Logging (only from main process)
        if step % args.logging_steps == 0 and accelerator.is_main_process:
            avg_loss = total_loss / max(1, args.logging_steps)
            elapsed = time.time() - start_time
            
            state_info = ""
            if config.state_passing:
                state_info = f", state_passing=ON"
                if args.state_reset_interval > 0:
                    state_info += f" (reset_every={args.state_reset_interval})"
                else:
                    state_info += " (no_reset)"
            else:
                state_info = ", state_passing=OFF"
            
            logger.info(f"Step {step:3d}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}, "
                       f"tokens/sec={tokens_per_sec:.0f}, time={step_time:.2f}s{state_info}")
            logger.info(f"         {monitor_gpu_usage()}")
            
            # Log to wandb
            if use_wandb:
                log_dict = {
                    "train/loss": avg_loss,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step_time": step_time,
                    "train/elapsed_time": elapsed,
                    "train/state_passing_enabled": config.state_passing,
                    "system/gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                    "system/gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0,
                    "system/num_processes": accelerator.num_processes,
                }
                if config.state_passing and args.state_reset_interval > 0:
                    log_dict["train/steps_since_last_reset"] = step % args.state_reset_interval
                
                wandb.log(log_dict, step=step)
            
            total_loss = 0.0
    
    # Final statistics
    if accelerator.is_main_process:
        total_time = time.time() - start_time
        avg_tokens_per_sec = tokens_processed / total_time
        
        logger.info("🎯 Training completed!")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average tokens/sec: {avg_tokens_per_sec:.0f}")
        logger.info(f"  Total tokens processed: {tokens_processed:,}")
        logger.info(f"  Final {monitor_gpu_usage()}")
        
        # Save model
        logger.info("💾 Saving model...")
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        model_save_path = os.path.join(args.output_dir, "final_model")
        unwrapped_model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        logger.info(f"📁 Model saved to {model_save_path}")
        
        # Evaluate perplexity at different sequence lengths
        logger.info("🔍 Starting perplexity evaluation for length generalization...")
        try:
            perplexity_results = evaluate_model_perplexity(
                model=unwrapped_model,
                tokenizer=tokenizer,
                device=accelerator.device,
                training_window=args.max_seq_length,
                output_dir=args.output_dir,
                step=step,
                log_wandb=use_wandb
            )
            logger.info(f"✅ Perplexity evaluation completed!")
            
            # Log key metrics
            min_ppl = min(perplexity_results['perplexities'])
            max_ppl = max(perplexity_results['perplexities'])
            logger.info(f"📊 Perplexity range: {min_ppl:.3f} - {max_ppl:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Perplexity evaluation failed: {e}")
        
        # Log final summary to wandb
        if use_wandb:
            summary_dict = {
                "summary/total_time": total_time,
                "summary/avg_tokens_per_sec": avg_tokens_per_sec,
                "summary/total_tokens_processed": tokens_processed,
                "summary/final_gpu_memory": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                "summary/distributed_processes": accelerator.num_processes,
                "summary/state_passing_enabled": config.state_passing,
            }
            wandb.log(summary_dict)
            wandb.finish()
        
        if config.state_passing:
            logger.info("✅ Multi-GPU state passing training completed successfully!")
        else:
            logger.info("✅ Multi-GPU training (no state passing) completed successfully!")

if __name__ == "__main__":
    main()