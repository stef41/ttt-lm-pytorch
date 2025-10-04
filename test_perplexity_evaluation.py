#!/usr/bin/env python3
"""
Test script for perplexity evaluation functionality.
"""

import torch
import logging
from transformers import AutoTokenizer
from ttt import TTTForCausalLM
from perplexity_evaluator import evaluate_model_perplexity
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_perplexity_evaluation():
    """Test the perplexity evaluation on a trained model."""
    
    # Use the most recent trained model
    model_path = "./outputs/full_dataset_training/final_model"
    
    logger.info(f"🧪 Testing perplexity evaluation with model: {model_path}")
    
    try:
        # Load model and tokenizer
        logger.info("📥 Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = TTTForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        logger.info(f"✅ Model loaded on device: {device}")
        
        # Run perplexity evaluation
        logger.info("🔍 Starting perplexity evaluation...")
        
        results = evaluate_model_perplexity(
            model=model,
            tokenizer=tokenizer,
            device=device,
            training_window=64,  # Updated to match new default training window
            output_dir="./test_perplexity_output",
            step=None,
            log_wandb=False  # Don't log to wandb for test
        )
        
        logger.info("✅ Perplexity evaluation completed successfully!")
        
        # Print results
        print(f"\n{'='*60}")
        print(f"PERPLEXITY EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Training window: {results['training_window']}")
        print(f"Evaluation lengths: {min(results['lengths'])} - {max(results['lengths'])}")
        print(f"Number of length points: {len(results['lengths'])}")
        print(f"Perplexity range: {min(results['perplexities']):.3f} - {max(results['perplexities']):.3f}")
        
        print(f"\n📊 Length vs Perplexity:")
        for length, ppl in zip(results['lengths'], results['perplexities']):
            ratio = length / results['training_window']
            print(f"  Length {length:4d} ({ratio:4.1f}x): Perplexity {ppl:7.3f}")
        
        print(f"\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test perplexity evaluation")
    parser.add_argument("--model_path", type=str, 
                       default="./outputs/full_dataset_training/final_model",
                       help="Path to the trained model")
    
    args = parser.parse_args()
    
    print(f"🧪 Testing Perplexity Evaluation System")
    print(f"Model: {args.model_path}")
    
    success = test_perplexity_evaluation()
    
    if success:
        print(f"\n🎉 All tests passed! Perplexity evaluation is ready to use.")
    else:
        print(f"\n💥 Tests failed! Please check the error messages above.")

if __name__ == "__main__":
    main()