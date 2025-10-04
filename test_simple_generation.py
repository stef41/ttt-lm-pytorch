#!/usr/bin/env python3
"""
Simple streaming test for TTT model that respects the mini-batch architecture.
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
    logger.info(f"Mini-batch size: {model.config.mini_batch_size}")
    
    return model, tokenizer, device

def simple_generate(model, tokenizer, device, prompt, max_new_tokens=30, temperature=0.8):
    """Generate text using the model's built-in generation (if available) or simple forward passes."""
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    logger.info(f"Starting generation with prompt: '{prompt}'")
    logger.info(f"Input length: {input_ids.shape[1]} tokens")
    
    print(f"\n{'='*50}")
    print(f"TTT MODEL GENERATION")
    print(f"{'='*50}")
    print(f"Prompt: {prompt}")
    print(f"Generated text: ", end="", flush=True)
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            try:
                # Forward pass with full sequence
                outputs = model(input_ids=generated_ids, use_cache=False)
                
                # Get next token logits
                next_token_logits = outputs.logits[0, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Decode and print token
                next_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(next_token_text, end="", flush=True)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Check for end of sequence
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
                # Small delay for streaming effect
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error during generation at step {step}: {e}")
                break
    
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    
    # Decode full generated sequence
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return full_text

def test_multiple_prompts(model, tokenizer, device):
    """Test the model with multiple different prompts."""
    
    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In the year 2025",
        "Machine learning is",
        "The secret to happiness"
    ]
    
    print(f"\n{'='*60}")
    print(f"TESTING MULTIPLE PROMPTS")
    print(f"{'='*60}")
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i}/5 ---")
        try:
            result = simple_generate(model, tokenizer, device, prompt, max_new_tokens=20, temperature=0.7)
            results.append((prompt, result))
            print(f"\nFull result: {result}")
        except Exception as e:
            logger.error(f"Failed on prompt '{prompt}': {e}")
            results.append((prompt, f"ERROR: {e}"))
    
    return results

def test_batch_generation(model, tokenizer, device):
    """Test batch generation with mini-batch size."""
    
    print(f"\n{'='*60}")
    print(f"TESTING BATCH GENERATION")
    print(f"{'='*60}")
    
    # Create a batch that matches mini-batch size
    mini_batch_size = model.config.mini_batch_size
    prompts = [
        "The weather is",
        "Today I will",
        "The future of AI",
        "In the garden",
        "Science tells us"
    ]
    
    # Repeat prompts to fill mini-batch if needed
    while len(prompts) < mini_batch_size:
        prompts.extend(prompts[:min(len(prompts), mini_batch_size - len(prompts))])
    
    prompts = prompts[:mini_batch_size]  # Trim to exact size
    
    logger.info(f"Testing with {len(prompts)} prompts (mini-batch size: {mini_batch_size})")
    
    # Tokenize all prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            print(f"✅ Batch forward pass successful!")
            print(f"Input shape: {inputs['input_ids'].shape}")
            print(f"Output logits shape: {logits.shape}")
            
            # Generate next tokens for each prompt in batch
            next_token_logits = logits[:, -1, :]  # Last position for each sequence
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            print(f"\nNext tokens for each prompt:")
            for i, (prompt, next_token) in enumerate(zip(prompts, next_tokens)):
                next_word = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(f"{i+1}. '{prompt}' → '{next_word}'")
                
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        print(f"❌ Batch generation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test TTT model generation")
    parser.add_argument("--model_path", type=str, 
                       default="./outputs/full_dataset_training/final_model",
                       help="Path to the trained model")
    parser.add_argument("--prompt", type=str,
                       default="The weather today is very nice, so I decided to",
                       help="Prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=30,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(args.model_path)
    
    print(f"\n🚀 Testing TTT Model Generation")
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Mini-batch size: {model.config.mini_batch_size}")
    
    # Test 1: Simple generation
    try:
        result = simple_generate(model, tokenizer, device, args.prompt, args.max_tokens, args.temperature)
        print(f"\n✅ Simple generation successful!")
    except Exception as e:
        logger.error(f"Simple generation failed: {e}")
    
    # Test 2: Multiple prompts
    try:
        results = test_multiple_prompts(model, tokenizer, device)
        print(f"\n✅ Multiple prompt test completed!")
    except Exception as e:
        logger.error(f"Multiple prompt test failed: {e}")
    
    # Test 3: Batch generation
    try:
        test_batch_generation(model, tokenizer, device)
    except Exception as e:
        logger.error(f"Batch generation test failed: {e}")
    
    print(f"\n✅ All tests completed!")

if __name__ == "__main__":
    main()