#!/usr/bin/env python3
"""
Test text generation with the properly trained TTT model.
"""

import torch
import sys
import argparse
from pathlib import Path
from transformers import AutoTokenizer

# Add the current directory to the path to import ttt
sys.path.append(str(Path(__file__).parent))
from ttt import TTTForCausalLM, TTTConfig

def test_generation(model_path="./properly_trained_model/final_model", prompt=None, max_length=50, temperature=0.8):
    print("🧪 Testing TTT Model Text Generation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    
    try:
        model = TTTForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model.to(device)
        model.eval()
        
        print(f"✓ Loaded trained model from {model_path}")
        print(f"✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Check convolution settings
        config = model.config
        print(f"✓ Convolutions disabled: {getattr(config, 'disable_conv', False)}")
        print(f"✓ Pre-conv disabled: {not getattr(config, 'pre_conv', True)}")
        print(f"✓ State passing enabled: {getattr(config, 'state_passing', False)}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test prompts
    if prompt is not None:
        test_prompts = [prompt]
    else:
        test_prompts = [
            "The capital of France is",
            "In the year 2030,",
            "Machine learning is",
            "The quick brown fox",
            "Hello world, this is"
        ]
    
    print(f"\n🔤 Text Generation Tests:")
    print("-" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate text manually (since TTTForCausalLM doesn't have generate method)
            input_ids = inputs.input_ids
            generated_ids = input_ids.clone()
            
            with torch.no_grad():
                for _ in range(15):  # Generate 15 tokens
                    # Forward pass
                    outputs = model(generated_ids)
                    logits = outputs.logits
                    
                    # Get next token probabilities
                    next_token_logits = logits[0, -1, :] / 0.8  # temperature
                    
                    # Apply top-k and top-p sampling
                    # Top-k
                    top_k = 50
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply softmax and sample
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Add to sequence
                    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                    
                    # Stop if EOS token
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            # Decode output
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated_part = generated_text[len(prompt):]
            
            print(f"\nTest {i}: '{prompt}'")
            print(f"Generated: '{generated_part}'")
            print(f"Full text: '{generated_text}'")
            
        except Exception as e:
            print(f"\nTest {i}: '{prompt}' - FAILED: {e}")
    
    print("\n" + "="*60)
    print("🎉 TEXT GENERATION TEST COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./properly_trained_model/final_model", help="Path to trained model")
    parser.add_argument("--prompt", default=None, help="Single prompt to test")
    parser.add_argument("--max_length", type=int, default=50, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    
    args = parser.parse_args()
    test_generation(args.model_path, args.prompt, args.max_length, args.temperature)