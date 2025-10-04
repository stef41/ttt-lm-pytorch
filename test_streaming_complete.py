#!/usr/bin/env python3
"""
Proper streaming test for TTT model using HuggingFace generation capabilities.
"""

import torch
import logging
from transformers import AutoTokenizer, TextStreamer
from ttt import TTTForCausalLM
import time
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_generation_methods(model, tokenizer, device, prompt, max_new_tokens=50):
    """Test different generation methods with the TTT model."""
    
    print(f"\n{'='*60}")
    print(f"TESTING GENERATION METHODS")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Method 1: Manual token-by-token generation (works reliably)
    print(f"\n--- Method 1: Manual Token-by-Token Generation ---")
    manual_result = manual_generate(model, tokenizer, device, input_ids, max_new_tokens)
    
    # Method 2: Try using built-in generate with custom parameters
    print(f"\n--- Method 2: Built-in Generate (if available) ---")
    try:
        with torch.no_grad():
            # Try the model's generate method
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            builtin_result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Built-in generate result: {builtin_result}")
    except Exception as e:
        print(f"Built-in generate failed: {e}")
        builtin_result = None
    
    # Method 3: Streaming with custom callback
    print(f"\n--- Method 3: Streaming Generation ---")
    streaming_result = streaming_generate(model, tokenizer, device, input_ids, max_new_tokens)
    
    return {
        'manual': manual_result,
        'builtin': builtin_result,
        'streaming': streaming_result
    }

def manual_generate(model, tokenizer, device, input_ids, max_new_tokens):
    """Manual token-by-token generation."""
    
    generated_ids = input_ids.clone()
    
    print("Manual generation: ", end="", flush=True)
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(input_ids=generated_ids)
            next_token_logits = outputs.logits[0, -1, :]
            
            # Sample with temperature
            probs = torch.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode and print
            next_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(next_token_text, end="", flush=True)
            
            # Append token
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    print()  # New line
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return result

def streaming_generate(model, tokenizer, device, input_ids, max_new_tokens):
    """Streaming generation with real-time display."""
    
    print("Streaming generation: ", end="", flush=True)
    
    generated_ids = input_ids.clone()
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(input_ids=generated_ids)
            next_token_logits = outputs.logits[0, -1, :]
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits / 0.8, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode and display with streaming effect
            next_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(next_token_text, end="", flush=True)
            time.sleep(0.1)  # Streaming delay
            
            # Update sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    print()  # New line
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return result

def test_conversation_mode(model, tokenizer, device):
    """Test conversation-like interaction with the model."""
    
    print(f"\n{'='*60}")
    print(f"CONVERSATION MODE TEST")
    print(f"{'='*60}")
    
    conversation = []
    context = ""
    
    prompts = [
        "Hello, how are you?",
        "What is your favorite color?",
        "Tell me about artificial intelligence.",
        "What did we just discuss?"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Human: {prompt}")
        
        # Build context
        if context:
            full_prompt = f"{context}\nHuman: {prompt}\nAssistant:"
        else:
            full_prompt = f"Human: {prompt}\nAssistant:"
        
        # Generate response
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        
        print("Assistant: ", end="", flush=True)
        
        with torch.no_grad():
            generated_ids = input_ids.clone()
            response_tokens = []
            
            for step in range(30):  # Limit response length
                outputs = model(input_ids=generated_ids)
                next_token_logits = outputs.logits[0, -1, :]
                
                probs = torch.softmax(next_token_logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop on newline or EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                next_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                
                # Stop on double newline (end of response)
                if '\n' in next_token_text and len(response_tokens) > 5:
                    break
                
                print(next_token_text, end="", flush=True)
                response_tokens.append(next_token.item())
                
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                time.sleep(0.05)
        
        print()  # New line
        
        # Update context for next turn
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        context = f"{full_prompt} {response_text}"
        conversation.append((prompt, response_text))
    
    print(f"\n--- Conversation Summary ---")
    for i, (human, assistant) in enumerate(conversation, 1):
        print(f"{i}. Human: {human}")
        print(f"   Assistant: {assistant}")

def main():
    parser = argparse.ArgumentParser(description="Test TTT model streaming generation")
    parser.add_argument("--model_path", type=str, 
                       default="./outputs/full_dataset_training/final_model",
                       help="Path to the trained model")
    parser.add_argument("--prompt", type=str,
                       default="The future of artificial intelligence",
                       help="Prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=40,
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    logger.info(f"Loading model from: {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = TTTForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"\n🚀 TTT Model Streaming Tests")
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Model type: {type(model).__name__}")
    
    # Test different generation methods
    results = test_generation_methods(model, tokenizer, device, args.prompt, args.max_tokens)
    
    # Test conversation mode
    test_conversation_mode(model, tokenizer, device)
    
    print(f"\n✅ All streaming tests completed!")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"GENERATION RESULTS SUMMARY")
    print(f"{'='*60}")
    for method, result in results.items():
        if result:
            print(f"{method.upper()}: {result[:100]}{'...' if len(result) > 100 else ''}")

if __name__ == "__main__":
    main()