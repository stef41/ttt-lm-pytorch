#!/usr/bin/env python3
"""
Test script to verify wandb online logging is working correctly.
"""

import wandb
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_wandb_online():
    """Test wandb online mode."""
    
    logger.info("🧪 Testing wandb online mode...")
    
    # Initialize wandb in online mode
    run = wandb.init(
        project="ttt-test",
        name="test-online-mode",
        mode="online",  # Force online mode
        config={
            "test_parameter": "online_mode_test",
            "timestamp": time.time()
        }
    )
    
    logger.info(f"✅ Wandb initialized successfully!")
    logger.info(f"   Run ID: {run.id}")
    logger.info(f"   Run URL: {run.get_url()}")
    logger.info(f"   Project: {run.project}")
    logger.info(f"   Mode: online")
    
    # Log some test metrics
    for step in range(5):
        wandb.log({
            "test_loss": 1.0 / (step + 1),
            "test_accuracy": step * 0.2,
            "step": step
        })
        logger.info(f"   Logged step {step}")
        time.sleep(1)
    
    logger.info("🎯 Test metrics logged successfully!")
    
    # Finish the run
    wandb.finish()
    logger.info("✅ Wandb run finished!")

def test_wandb_offline():
    """Test wandb offline mode for comparison."""
    
    logger.info("🧪 Testing wandb offline mode...")
    
    # Initialize wandb in offline mode
    run = wandb.init(
        project="ttt-test",
        name="test-offline-mode",
        mode="offline",  # Force offline mode
        config={
            "test_parameter": "offline_mode_test",
            "timestamp": time.time()
        }
    )
    
    logger.info(f"✅ Wandb initialized successfully!")
    logger.info(f"   Run ID: {run.id}")
    logger.info(f"   Project: {run.project}")
    logger.info(f"   Mode: offline")
    
    # Log some test metrics
    for step in range(3):
        wandb.log({
            "test_loss": 2.0 / (step + 1),
            "test_accuracy": step * 0.15,
            "step": step
        })
        logger.info(f"   Logged step {step}")
        time.sleep(0.5)
    
    logger.info("🎯 Test metrics logged successfully!")
    
    # Finish the run
    wandb.finish()
    logger.info("✅ Wandb run finished!")

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"WANDB ONLINE MODE TEST")
    print(f"{'='*60}")
    
    try:
        # Test online mode
        test_wandb_online()
        
        print(f"\n{'='*60}")
        print(f"WANDB OFFLINE MODE TEST (for comparison)")
        print(f"{'='*60}")
        
        # Test offline mode for comparison
        test_wandb_offline()
        
        print(f"\n{'='*60}")
        print(f"✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"Wandb online logging is now configured by default.")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print(f"\n{'='*60}")
        print(f"❌ TEST FAILED!")
        print(f"Error: {e}")
        print(f"{'='*60}")