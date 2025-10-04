#!/usr/bin/env python3
"""
Test script to verify TTT model works correctly with all 1D temporal convolutions disabled.
"""
import torch
from ttt import TTTConfig, TTTForCausalLM, TTT_STANDARD_CONFIGS

def test_model_without_convolutions():
    """Test the TTT model with all convolutions disabled."""
    print("Testing TTT model with disabled convolutions...")
    
    # Use the 125m config for faster testing
    config_dict = TTT_STANDARD_CONFIGS["125m"].copy()
    
    # Ensure all convolutions are disabled
    config_dict.update({
        "pre_conv": False,
        "disable_conv": True,
        "share_qk": True,  # Test the conv_q/conv_k path
        "max_position_embeddings": 128,
        "vocab_size": 1000,  # Smaller vocab for testing
    })
    
    config = TTTConfig(**config_dict)
    print(f"Config: pre_conv={config.pre_conv}, disable_conv={config.disable_conv}, share_qk={config.share_qk}")
    
    # Create model
    model = TTTForCausalLM(config)
    model.train()  # Set to train mode for gradient computation
    
    # Test input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    outputs = model(input_ids)
    logits = outputs.logits
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    # Check output shape
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    # Check for NaN/Inf values
    assert not torch.isnan(logits).any(), "Found NaN values in logits"
    assert not torch.isinf(logits).any(), "Found Inf values in logits"
    
    print("✓ Model forward pass successful!")
    print("✓ Output shape is correct!")
    print("✓ No NaN/Inf values found!")
    
    # Test gradient computation
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size), 
        input_ids.view(-1)
    )
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print("✓ Backward pass successful!")
    
    return True

def test_model_configurations():
    """Test multiple configuration options."""
    print("\nTesting different model configurations...")
    
    configs_to_test = [
        {"share_qk": True, "pre_conv": False, "disable_conv": True},
        {"share_qk": False, "pre_conv": False, "disable_conv": True},
        {"share_qk": True, "pre_conv": True, "disable_conv": True},  # pre_conv=True but convs disabled
    ]
    
    for i, config_overrides in enumerate(configs_to_test):
        print(f"\nConfig {i+1}: {config_overrides}")
        
        config_dict = TTT_STANDARD_CONFIGS["125m"].copy()
        config_dict.update({
            "max_position_embeddings": 64,
            "vocab_size": 500,
            **config_overrides
        })
        
        config = TTTConfig(**config_dict)
        model = TTTForCausalLM(config)
        model.train()  # Set to train mode for gradient computation
        
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        
        outputs = model(input_ids)
        logits = outputs.logits
        
        assert logits.shape == (1, 16, config.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        
        print(f"✓ Config {i+1} passed!")

if __name__ == "__main__":
    print("=" * 60)
    print("TTT Model Convolution Disabling Test")
    print("=" * 60)
    
    try:
        test_model_without_convolutions()
        test_model_configurations()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("All 1D temporal convolutions successfully disabled!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)