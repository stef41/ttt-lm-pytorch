#!/usr/bin/env python3
"""
Debug script to trace the exact location of CUDA indexing errors in TTT model.
This will help identify which specific operation causes the "index out of bounds" error.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import logging
import traceback
import os
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_tensor_shapes_and_values(name, tensor):
    """Debug helper to print tensor info safely."""
    if tensor is None:
        logger.debug(f"{name}: None")
        return
    
    logger.debug(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    if tensor.numel() > 0:
        logger.debug(f"  min={tensor.min().item()}, max={tensor.max().item()}")
        if tensor.dtype in [torch.long, torch.int]:
            logger.debug(f"  unique values: {torch.unique(tensor[:10])}")  # First 10 to avoid spam

def create_debug_batch(tokenizer, seq_length=64, batch_size=2):
    """Create a small debug batch with known safe values."""
    # Create simple text that should tokenize cleanly
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test sentence for debugging."
    ]
    
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=seq_length,
        return_tensors="pt"
    )
    
    # Debug the tokenized inputs
    debug_tensor_shapes_and_values("input_ids", tokenized["input_ids"])
    debug_tensor_shapes_and_values("attention_mask", tokenized["attention_mask"])
    
    # Check for invalid token IDs
    vocab_size = tokenizer.vocab_size
    max_token_id = tokenized["input_ids"].max().item()
    min_token_id = tokenized["input_ids"].min().item()
    
    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Token ID range: {min_token_id} to {max_token_id}")
    
    if max_token_id >= vocab_size:
        logger.error(f"ERROR: Token ID {max_token_id} >= vocab_size {vocab_size}")
        return None
    
    if min_token_id < 0:
        logger.error(f"ERROR: Negative token ID {min_token_id}")
        return None
    
    # Create labels
    labels = tokenized["input_ids"].clone()
    labels[tokenized["attention_mask"] == 0] = -100
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

def debug_model_components(model, batch):
    """Step through model components to find where the error occurs."""
    logger.info("=== Debugging Model Components ===")
    
    try:
        # Step 1: Test embedding layer
        logger.info("Step 1: Testing embedding layer...")
        input_ids = batch["input_ids"]
        debug_tensor_shapes_and_values("input_ids", input_ids)
        
        embed_tokens = model.model.embed_tokens
        logger.debug(f"Embedding layer: vocab_size={embed_tokens.num_embeddings}, embed_dim={embed_tokens.embedding_dim}")
        
        # Test embedding lookup
        inputs_embeds = embed_tokens(input_ids)
        debug_tensor_shapes_and_values("inputs_embeds", inputs_embeds)
        logger.info("✅ Embedding layer OK")
        
    except Exception as e:
        logger.error(f"❌ Error in embedding layer: {e}")
        logger.error(traceback.format_exc())
        return False
    
    try:
        # Step 2: Test position_ids generation
        logger.info("Step 2: Testing position_ids generation...")
        
        seqlen_offset = 0
        seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device
        
        logger.debug(f"seqlen_offset={seqlen_offset}, seq_len={seq_len}")
        
        position_ids = torch.arange(
            seqlen_offset,
            seqlen_offset + seq_len,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        
        debug_tensor_shapes_and_values("position_ids", position_ids)
        logger.info("✅ Position IDs generation OK")
        
    except Exception as e:
        logger.error(f"❌ Error in position_ids generation: {e}")
        logger.error(traceback.format_exc())
        return False
    
    try:
        # Step 3: Test first layer forward pass
        logger.info("Step 3: Testing first layer...")
        
        first_layer = model.model.layers[0]
        logger.debug(f"First layer type: {type(first_layer)}")
        
        # Test with minimal inputs
        hidden_states = inputs_embeds
        attention_mask = batch["attention_mask"]
        
        debug_tensor_shapes_and_values("hidden_states (input)", hidden_states)
        debug_tensor_shapes_and_values("attention_mask", attention_mask)
        
        # Try forward pass through first layer
        layer_output = first_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=None
        )
        
        debug_tensor_shapes_and_values("layer_output", layer_output)
        logger.info("✅ First layer forward pass OK")
        
    except Exception as e:
        logger.error(f"❌ Error in first layer: {e}")
        logger.error(traceback.format_exc())
        return False
    
    return True

def debug_ttt_layers(model, batch):
    """Debug TTT-specific layers to find issues."""
    logger.info("=== Debugging TTT Layers ===")
    
    try:
        first_layer = model.model.layers[0]
        ttt_layer = first_layer.seq_modeling_block
        
        logger.debug(f"TTT layer type: {type(ttt_layer)}")
        
        # Check TTT layer components
        if hasattr(ttt_layer, 'W1'):
            if hasattr(ttt_layer.W1, 'weight'):
                debug_tensor_shapes_and_values("W1 weight", ttt_layer.W1.weight)
            else:
                debug_tensor_shapes_and_values("W1", ttt_layer.W1)
        if hasattr(ttt_layer, 'W2'):
            if hasattr(ttt_layer.W2, 'weight'):
                debug_tensor_shapes_and_values("W2 weight", ttt_layer.W2.weight)
            else:
                debug_tensor_shapes_and_values("W2", ttt_layer.W2)
        
        # Test TTT layer forward pass
        inputs_embeds = model.model.embed_tokens(batch["input_ids"])
        hidden_states = inputs_embeds
        
        # Generate position_ids like the model does
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        debug_tensor_shapes_and_values("position_ids for TTT", position_ids)
        
        logger.info("Testing TTT layer forward pass...")
        ttt_output = ttt_layer(
            hidden_states=hidden_states,
            attention_mask=batch["attention_mask"],
            position_ids=position_ids,
            cache_params=None
        )
        
        debug_tensor_shapes_and_values("ttt_output", ttt_output)
        logger.info("✅ TTT layer forward pass OK")
        
    except Exception as e:
        logger.error(f"❌ Error in TTT layer: {e}")
        logger.error(traceback.format_exc())
        return False
    
    return True

def main():
    logger.info("🔍 Starting TTT Debug Session")
    
    # Create model with conservative settings
    config = TTT_STANDARD_CONFIGS["125m"].copy()
    config["state_passing"] = False  # Disable to isolate issues
    config["max_position_embeddings"] = 512
    config = TTTConfig(**config)
    
    logger.info(f"Model config: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
    
    try:
        model = TTTForCausalLM(config)
        logger.info("✅ Model creation OK")
    except Exception as e:
        logger.error(f"❌ Error creating model: {e}")
        return
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(f"Model vocab size: {config.vocab_size}")
    
    if tokenizer.vocab_size != config.vocab_size:
        logger.warning(f"Vocab size mismatch! Tokenizer: {tokenizer.vocab_size}, Model: {config.vocab_size}")
    
    # Create debug batch
    batch = create_debug_batch(tokenizer, seq_length=32, batch_size=2)
    if batch is None:
        logger.error("Failed to create debug batch")
        return
    
    # Move to GPU if available
    if torch.cuda.is_available():
        logger.info("Moving to GPU...")
        model = model.cuda()
        batch = {k: v.cuda() for k, v in batch.items()}
    
    # Debug step by step
    model.eval()
    
    with torch.no_grad():
        # Test basic components
        if not debug_model_components(model, batch):
            logger.error("Basic component debugging failed")
            return
        
        # Test TTT specific layers
        if not debug_ttt_layers(model, batch):
            logger.error("TTT layer debugging failed")
            return
        
        # Test full forward pass
        logger.info("=== Testing Full Forward Pass ===")
        try:
            outputs = model(**batch)
            logger.info("✅ Full forward pass OK")
            logger.info(f"Loss: {outputs.loss.item():.4f}")
            
            debug_tensor_shapes_and_values("logits", outputs.logits)
            
        except Exception as e:
            logger.error(f"❌ Error in full forward pass: {e}")
            logger.error(traceback.format_exc())
    
    logger.info("🏁 Debug session complete")

if __name__ == "__main__":
    main()