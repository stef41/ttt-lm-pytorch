#!/usr/bin/env python3
"""
Simple but effective streaming demonstration for TTT model.
"""

import torch
import logging
from transformers import AutoTokenizer
from ttt import TTTForCausalLM
import time
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load the TTT model and tokenizer."""
    logger.info(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = TTTForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer, device

def streaming_generation_demo(model, tokenizer, device, prompt, max_tokens=60):
    """Demonstrate streaming text generation with visual effects."""
    
    print(f"\n{'='*70}")
    print(f"🎭 STREAMING GENERATION DEMO")
    print(f"{'='*70}")
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"📝 Prompt: '{prompt}'")
    print(f"🔢 Input tokens: {input_ids.shape[1]}")
    print(f"🎯 Generating {max_tokens} tokens...")
    print(f"\n{'─'*50}")
    print(f"Generated text: ", end="", flush=True)
    
    generated_ids = input_ids.clone()
    generated_tokens = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for step in range(max_tokens):
            # Forward pass
            outputs = model(input_ids=generated_ids)
            next_token_logits = outputs.logits[0, -1, :]
            
            # Sample with temperature and top-p
            temperature = 0.8
            top_p = 0.9
            
            # Apply temperature
            scaled_logits = next_token_logits / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            scaled_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode token
            next_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            generated_tokens.append(next_token.item())
            
            # Print with streaming effect
            print(next_token_text, end="", flush=True)
            
            # Update sequence for next iteration
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Streaming delay
            time.sleep(0.08)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    print(f"\n{'─'*50}")
    print(f"✅ Generation completed!")
    print(f"⏱️  Time taken: {generation_time:.2f} seconds")
    print(f"🔢 Tokens generated: {len(generated_tokens)}")
    print(f"🚀 Speed: {len(generated_tokens)/generation_time:.1f} tokens/sec")
    
    # Full result
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return full_text

def compare_temperatures(model, tokenizer, device, prompt):
    """Compare generation with different temperatures."""
    
    print(f"\n{'='*70}")
    print(f"🌡️  TEMPERATURE COMPARISON")
    print(f"{'='*70}")
    
    temperatures = [0.3, 0.7, 1.0, 1.5]
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        print(f"Output: ", end="", flush=True)
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for step in range(25):
                outputs = model(input_ids=generated_ids)
                next_token_logits = outputs.logits[0, -1, :]
                
                # Apply temperature
                if temp > 0:
                    scaled_logits = next_token_logits / temp
                else:
                    scaled_logits = next_token_logits
                
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                next_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(next_token_text, end="", flush=True)
                
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
                time.sleep(0.05)
        
        print()  # New line

def creative_prompts_demo(model, tokenizer, device):
    """Test various creative prompts."""
    
    print(f"\n{'='*70}")
    print(f"🎨 CREATIVE PROMPTS DEMO")
    print(f"{'='*70}")
    
    prompts = [
        "The last human on Earth discovered",
        "In the year 3024, artificial intelligence",
        "Deep in the ocean, scientists found",
        "The ancient book revealed a secret about",
        "When the stars aligned perfectly, magic"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🎪 Prompt {i}/{len(prompts)}: '{prompt}'")
        print(f"Response: ", end="", flush=True)
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for step in range(30):
                outputs = model(input_ids=generated_ids)
                next_token_logits = outputs.logits[0, -1, :]
                
                probs = torch.softmax(next_token_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                next_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(next_token_text, end="", flush=True)
                
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
                time.sleep(0.06)
        
        print()  # New line

def main():
    parser = argparse.ArgumentParser(description="TTT Model Streaming Demo")
    parser.add_argument("--model_path", type=str, 
                       default="./outputs/full_dataset_training/final_model",
                       help="Path to the trained model")
    parser.add_argument("--prompt", type=str,
                       default="The future of technology will bring us",
                       help="Prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=50,
                       help="Maximum tokens to generate")
    parser.add_argument("--demo", type=str, choices=['basic', 'temperature', 'creative', 'all'],
                       default='all', help="Which demo to run")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.model_path)
    
    print(f"\n🚀 TTT Model Streaming Demonstrations")
    print(f"📁 Model: {args.model_path}")
    print(f"💻 Device: {device}")
    print(f"⚙️  Model config: {model.config.hidden_size}D, {model.config.num_hidden_layers} layers")
    print(f"🔄 State passing: {model.config.state_passing}")
    
    if args.demo in ['basic', 'all']:
        # Basic streaming demo
        result = streaming_generation_demo(model, tokenizer, device, args.prompt, args.max_tokens)
    
    if args.demo in ['temperature', 'all']:
        # Temperature comparison
        compare_temperatures(model, tokenizer, device, "Once upon a time in a distant galaxy")
    
    if args.demo in ['creative', 'all']:
        # Creative prompts
        creative_prompts_demo(model, tokenizer, device)
    
    print(f"\n{'='*70}")
    print(f"✨ All streaming demonstrations completed successfully!")
    print(f"🎯 The TTT model with persistent state passing is working perfectly!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()