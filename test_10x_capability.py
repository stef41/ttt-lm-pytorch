#!/usr/bin/env python3
"""
Quick test to verify 10x evaluation capability
"""

from perplexity_evaluator import PerplexityEvaluator
from transformers import AutoTokenizer
from ttt import TTTForCausalLM
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_10x_capability():
    """Test if we can evaluate up to 10x the training window"""
    print("🧪 Testing 10x evaluation capability")
    
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
    
    # Initialize evaluator
    training_window = 64
    max_eval_length = training_window * 10  # 640 tokens
    evaluator = PerplexityEvaluator(model, tokenizer, device, max_eval_length)
    
    # Test specific long lengths
    test_lengths = [
        training_window * 2,   # 256 tokens (2x)
        training_window * 5,   # 640 tokens (5x)
        training_window * 8,   # 1024 tokens (8x)
        training_window * 10,  # 640 tokens (10x)
    ]
    
    print("🔍 Testing specific long lengths...")
    
    # Prepare dataset
    eval_texts = evaluator.prepare_eval_dataset(num_samples=100)
    print(f"📊 Prepared {len(eval_texts)} evaluation texts")
    
    results = {}
    
    for length in test_lengths:
        ratio = length / training_window
        print(f"\n🎯 Testing {length} tokens ({ratio:.1f}x training window)...")
        
        try:
            # Get sequences at this length
            sequences = evaluator.tokenize_at_length(eval_texts, length)
            print(f"   Found {len(sequences)} sequences of length {length}")
            
            if len(sequences) >= 3:
                # Compute perplexity
                perplexity = evaluator.compute_perplexity_at_length(sequences)
                results[length] = perplexity
                print(f"   ✅ Perplexity: {perplexity:.3f}")
            else:
                print(f"   ⚠️  Not enough sequences (only {len(sequences)})")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n🎉 Results Summary:")
    print(f"📊 Successfully evaluated {len(results)} different lengths")
    
    max_length_tested = max(results.keys()) if results else 0
    max_ratio_tested = max_length_tested / training_window if results else 0
    
    print(f"🏆 Maximum length successfully tested: {max_length_tested} tokens ({max_ratio_tested:.1f}x)")
    
    if max_length_tested >= training_window * 10:
        print("🎯 SUCCESS: Achieved 10x evaluation capability!")
    elif max_length_tested >= training_window * 5:
        print("🎯 PARTIAL SUCCESS: Achieved at least 5x evaluation capability!")
    else:
        print("🎯 LIMITED: Could only achieve shorter sequence evaluation")
    
    return results

if __name__ == "__main__":
    test_10x_capability()