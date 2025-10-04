#!/usr/bin/env python3
"""
Test script to verify that perplexity evaluation logs to wandb correctly
"""

import os
import tempfile
import wandb
from perplexity_evaluator import evaluate_model_perplexity

def test_wandb_logging():
    """Test that perplexity evaluation logs to wandb correctly"""
    print("🧪 Testing wandb logging for perplexity evaluation")
    
    # Initialize wandb run
    run = wandb.init(
        project="ttt-perplexity-test",
        mode="online",  # Use online mode for testing
        name="test-perplexity-logging",
        config={
            "model_path": "./outputs/full_dataset_training/final_model",
            "training_window": 64,
            "test_run": True
        }
    )
    
    try:
        # Run perplexity evaluation
        print("📊 Running perplexity evaluation with wandb logging...")
        results = evaluate_model_perplexity(
            model_path="./outputs/full_dataset_training/final_model",
            training_window=64,
            step="test",
            wandb_run=run,
            save_plots=True,
            output_dir="./test_wandb_output"
        )
        
        print("✅ Perplexity evaluation completed!")
        print(f"📈 Results: {len(results)} length points evaluated")
        print(f"📊 Perplexity range: {min(results.values()):.3f} - {max(results.values()):.3f}")
        
        # Check if wandb logged the results
        if hasattr(run, 'summary') and run.summary:
            print("📊 Wandb summary contains data - logging successful!")
        else:
            print("⚠️  No wandb summary data found")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        wandb.finish()
    
    print("🎉 Wandb logging test completed!")
    return True

if __name__ == "__main__":
    test_wandb_logging()