#!/usr/bin/env python3
"""
Simple test to verify perplexity evaluation integration is working
"""

from perplexity_evaluator import evaluate_model_perplexity
from transformers import AutoTokenizer
from ttt import TTTForCausalLM
import torch

def test_evaluation_integration():
    """Test the basic integration without wandb"""
    print("🧪 Testing perplexity evaluation integration")
    
    # Load model and tokenizer
    model_path = "./outputs/full_dataset_training/final_model"
    
    print("📥 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = TTTForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"✅ Model loaded on device: {device}")
    
    print("🔍 Testing perplexity evaluation (basic, no wandb)...")
    try:
        results = evaluate_model_perplexity(
            model=model,
            tokenizer=tokenizer,
            device=device,
            training_window=64,
            step=None,  # No step for this test
            log_wandb=False  # No wandb for this test
        )
        
        print(f"✅ Evaluation completed successfully!")
        print(f"📊 Evaluated {len(results['lengths'])} different sequence lengths")
        if results['perplexities']:
            min_perplexity = min(results['perplexities'])
            max_perplexity = max(results['perplexities'])
            print(f"📈 Perplexity range: {min_perplexity:.3f} - {max_perplexity:.3f}")
            print(f"📏 Length range: {min(results['lengths'])} - {max(results['lengths'])}")
        
        print("🎉 Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_evaluation_integration()