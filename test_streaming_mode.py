#!/usr/bin/env python3
"""
Test TTT model in streaming mode to demonstrate persistent state passing.
This script shows how the model maintains memory across token generation.
"""

import torch
import logging
from transformers import AutoTokenizer
from ttt import TTTForCausalLM, TTTConfig
import time
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path):
    """Load the trained TTT model and tokenizer."""
    logger.info(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = TTTForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    logger.info(f"Model loaded on device: {device}")
    logger.info(f"Model config: {model.config}")
    
    return model, tokenizer, device

def generate_streaming(model, tokenizer, device, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    """Generate tokens in streaming mode, showing each token as it's generated."""
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    logger.info(f"Starting generation with prompt: '{prompt}'")
    logger.info(f"Input length: {input_ids.shape[1]} tokens")
    
    # Initialize generation
    generated_ids = input_ids.clone()
    
    print(f"\n{'='*50}")
    print(f"STREAMING GENERATION (Persistent State)")
    print(f"{'='*50}")
    print(f"Prompt: {prompt}")
    print(f"Generated text: ", end="", flush=True)
    
    # Track state persistence
    cache_params = None
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass with current sequence
            outputs = model(
                input_ids=generated_ids,
                cache_params=cache_params,
                use_cache=True
            )
            
            # Update cache for next iteration (persistent state)
            cache_params = outputs.cache_params
            
            # Get next token logits
            next_token_logits = outputs.logits[0, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
                # Apply top-p sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode and print token
            next_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(next_token_text, end="", flush=True)
            
            # Update generated sequence (only the new token for next iteration)
            generated_ids = next_token.unsqueeze(0)
            
            # Check for end of sequence
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # Small delay to show streaming effect
            time.sleep(0.05)
    
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    
    return cache_params

def test_context_continuity(model, tokenizer, device):
    """Test that the model maintains context across multiple generations."""
    
    print(f"\n{'='*60}")
    print(f"TESTING CONTEXT CONTINUITY")
    print(f"{'='*60}")
    
    # Start with a context
    context = "The ancient library contained many secrets. Among the dusty shelves"
    
    # Generate continuation 1
    print(f"\n--- First Generation ---")
    cache_params = generate_streaming(model, tokenizer, device, context, max_new_tokens=20)
    
    # Continue with the persistent state
    print(f"\n--- Second Generation (continuing with persistent state) ---")
    continuation = ", and behind the old oak door"
    
    # Encode continuation
    input_ids = tokenizer.encode(continuation, return_tensors="pt").to(device)
    
    print(f"Continuing with: '{continuation}'")
    print(f"Generated text: ", end="", flush=True)
    
    with torch.no_grad():
        for step in range(20):
            outputs = model(
                input_ids=input_ids,
                cache_params=cache_params,
                use_cache=True
            )
            
            cache_params = outputs.cache_params
            next_token_logits = outputs.logits[0, -1, :]
            
            # Simple sampling
            probs = torch.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            next_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(next_token_text, end="", flush=True)
            
            input_ids = next_token.unsqueeze(0)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            time.sleep(0.05)
    
    print(f"\n{'='*60}")
    print("Context continuity test complete!")

def main():
    parser = argparse.ArgumentParser(description="Test TTT model in streaming mode")
    parser.add_argument("--model_path", type=str, 
                       default="./outputs/full_dataset_training/final_model",
                       help="Path to the trained model")
    parser.add_argument("--prompt", type=str,
                       default="Once upon a time in a magical forest",
                       help="Prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=50,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(args.model_path)
    
    # Test basic streaming generation
    print(f"\n🚀 Testing TTT Model in Streaming Mode")
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    
    # Single generation test
    generate_streaming(
        model, tokenizer, device, 
        args.prompt, 
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Test context continuity with persistent state
    test_context_continuity(model, tokenizer, device)
    
    print(f"\n✅ Streaming mode test completed successfully!")

if __name__ == "__main__":
    main()