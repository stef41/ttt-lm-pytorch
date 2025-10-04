#!/usr/bin/env python3
"""
Demonstrate TTT model's persistent state capabilities in streaming mode.
This shows how the model maintains context across multiple generation calls.
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

def demo_persistent_state(model, tokenizer, device):
    """Demonstrate persistent state across multiple generations."""
    
    print(f"\n{'='*70}")
    print(f"PERSISTENT STATE DEMONSTRATION")
    print(f"{'='*70}")
    print("This demo shows how TTT maintains context across multiple calls")
    print("Each generation builds on the previous state")
    
    # Story segments to build progressively
    story_parts = [
        "Once upon a time, in a mystical forest",
        "there lived a wise old wizard who",
        "possessed ancient knowledge of magic and",
        "taught young apprentices the secrets of",
        "spellcasting and potion making, until one day"
    ]
    
    cache_params = None
    full_story = ""
    
    for i, part in enumerate(story_parts, 1):
        print(f"\n--- Part {i}/5: Building the Story ---")
        print(f"Input: '{part}'")
        
        # Encode current part
        input_ids = tokenizer.encode(part, return_tensors="pt").to(device)
        
        print(f"Generated continuation: ", end="", flush=True)
        
        with torch.no_grad():
            generated_ids = input_ids.clone()
            
            for step in range(15):  # Generate 15 tokens per part
                outputs = model(
                    input_ids=generated_ids,
                    cache_params=cache_params,
                    use_cache=True
                )
                
                # Update cache for persistent state
                cache_params = outputs.cache_params
                
                # Get next token
                next_token_logits = outputs.logits[0, -1, :]
                probs = torch.softmax(next_token_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Decode and display
                next_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(next_token_text, end="", flush=True)
                
                # For next iteration, only use the new token
                generated_ids = next_token.unsqueeze(0)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
                time.sleep(0.1)
        
        print(f"\n")
        
        # Build full story for display
        current_continuation = tokenizer.decode(
            outputs.logits.argmax(dim=-1)[0][-15:], 
            skip_special_tokens=True
        )
        full_story += f"{part} "
    
    print(f"\n{'='*70}")
    print(f"DEMONSTRATION COMPLETE")
    print(f"{'='*70}")
    print(f"The model successfully maintained persistent state across {len(story_parts)} parts!")
    print(f"Cache was never reset, allowing continuous context building.")

def interactive_streaming_chat(model, tokenizer, device):
    """Interactive chat mode with streaming responses."""
    
    print(f"\n{'='*70}")
    print(f"INTERACTIVE STREAMING CHAT")
    print(f"{'='*70}")
    print("Type 'quit' to exit, 'reset' to clear context")
    
    cache_params = None
    conversation_history = ""
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            cache_params = None
            conversation_history = ""
            print("🔄 Context reset!")
            continue
        
        # Build prompt with conversation history
        if conversation_history:
            prompt = f"{conversation_history}\nHuman: {user_input}\nAssistant:"
        else:
            prompt = f"Human: {user_input}\nAssistant:"
        
        # Generate response
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        print("Assistant: ", end="", flush=True)
        
        response_tokens = []
        
        with torch.no_grad():
            generated_ids = input_ids.clone()
            
            for step in range(50):
                outputs = model(
                    input_ids=generated_ids,
                    cache_params=cache_params,
                    use_cache=True
                )
                
                # Update persistent state
                cache_params = outputs.cache_params
                
                next_token_logits = outputs.logits[0, -1, :]
                probs = torch.softmax(next_token_logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                next_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                
                # Stop on newlines or EOS
                if '\n' in next_token_text or next_token.item() == tokenizer.eos_token_id:
                    if len(response_tokens) > 3:  # Ensure minimum response length
                        break
                
                print(next_token_text, end="", flush=True)
                response_tokens.append(next_token.item())
                
                # For next iteration
                generated_ids = next_token.unsqueeze(0)
                time.sleep(0.05)
        
        print()  # New line
        
        # Update conversation history
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        conversation_history += f"\nHuman: {user_input}\nAssistant: {response_text}"

def main():
    parser = argparse.ArgumentParser(description="TTT Model Streaming Demo")
    parser.add_argument("--model_path", type=str, 
                       default="./outputs/full_dataset_training/final_model",
                       help="Path to the trained model")
    parser.add_argument("--mode", type=str, choices=['demo', 'chat', 'both'], 
                       default='both', help="Demo mode to run")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.model_path)
    
    print(f"\n🚀 TTT Model Streaming Demo")
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Config: state_passing={model.config.state_passing}")
    
    if args.mode in ['demo', 'both']:
        # Run persistent state demonstration
        demo_persistent_state(model, tokenizer, device)
    
    if args.mode in ['chat', 'both']:
        # Run interactive chat
        try:
            interactive_streaming_chat(model, tokenizer, device)
        except KeyboardInterrupt:
            print("\n\n👋 Chat ended by user")
    
    print(f"\n✅ Demo completed!")

if __name__ == "__main__":
    main()