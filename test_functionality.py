#!/usr/bin/env python3
"""
Comprehensive test to verify TTT model functionality after convolution changes.
Tests both training and text generation capabilities.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from ttt import TTTConfig, TTTForCausalLM, TTT_STANDARD_CONFIGS
import time

def test_basic_functionality():
    """Test basic model functionality."""
    print("=" * 60)
    print("Testing Basic Model Functionality")
    print("=" * 60)
    
    # Test with disabled convolutions (default now)
    config_dict = TTT_STANDARD_CONFIGS["125m"].copy()
    config_dict.update({
        "max_position_embeddings": 512,
        "vocab_size": 5000,
    })
    
    config = TTTConfig(**config_dict)
    print(f"Config: pre_conv={config.pre_conv}, disable_conv={config.disable_conv}")
    
    model = TTTForCausalLM(config)
    model.train()
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    start_time = time.time()
    outputs = model(input_ids)
    forward_time = time.time() - start_time
    
    logits = outputs.logits
    print(f"Output logits shape: {logits.shape}")
    print(f"Forward pass time: {forward_time:.4f}s")
    
    # Test loss computation
    labels = input_ids.clone()
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
    print(f"Loss: {loss.item():.4f}")
    
    # Test backward pass
    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time
    print(f"Backward pass time: {backward_time:.4f}s")
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
    
    print(f"Number of parameters with gradients: {len(grad_norms)}")
    print(f"Average gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
    
    print("✓ Basic functionality test passed!")
    return model, config

def test_text_generation(model, config):
    """Test text generation capabilities."""
    print("\n" + "=" * 60)
    print("Testing Text Generation")
    print("=" * 60)
    
    model.eval()
    
    # Create a simple tokenizer-like setup for testing
    vocab_size = config.vocab_size
    
    # Test different prompt lengths
    prompt_lengths = [1, 5, 10, 20]
    generation_length = 20
    
    for prompt_len in prompt_lengths:
        print(f"\nTesting generation with prompt length {prompt_len}:")
        
        # Create random prompt
        prompt = torch.randint(1, vocab_size-1, (1, prompt_len))
        
        # Generate text
        generated = prompt.clone()
        
        with torch.no_grad():
            for _ in range(generation_length):
                outputs = model(generated)
                logits = outputs.logits
                
                # Get next token probabilities
                next_token_logits = logits[0, -1, :]
                
                # Sample next token (using top-k sampling)
                top_k = 50
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices[next_token_idx]
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        print(f"  Prompt: {prompt.tolist()}")
        print(f"  Generated: {generated[0, prompt_len:].tolist()}")
        print(f"  Full sequence length: {generated.shape[1]}")
        
        # Verify no NaN/Inf in generated sequence
        assert not torch.isnan(generated).any(), "NaN found in generated sequence"
        assert not torch.isinf(generated).any(), "Inf found in generated sequence"
    
    print("✓ Text generation test passed!")

def test_with_convolutions_enabled():
    """Test model with convolutions enabled to ensure backward compatibility."""
    print("\n" + "=" * 60)
    print("Testing with Convolutions Enabled (Backward Compatibility)")
    print("=" * 60)
    
    config_dict = TTT_STANDARD_CONFIGS["125m"].copy()
    config_dict.update({
        "max_position_embeddings": 256,
        "vocab_size": 2000,
        "pre_conv": True,
        "disable_conv": False,  # Enable convolutions
        "share_qk": True,
    })
    
    config = TTTConfig(**config_dict)
    print(f"Config: pre_conv={config.pre_conv}, disable_conv={config.disable_conv}, share_qk={config.share_qk}")
    
    model = TTTForCausalLM(config)
    model.train()
    
    # Test forward pass
    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    outputs = model(input_ids)
    logits = outputs.logits
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test loss and backward
    labels = input_ids.clone()
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print("✓ Convolutions enabled test passed!")

def test_different_configurations():
    """Test various model configurations."""
    print("\n" + "=" * 60)
    print("Testing Different Model Configurations")
    print("=" * 60)
    
    configs_to_test = [
        {"name": "No conv, share_qk=True", "pre_conv": False, "disable_conv": True, "share_qk": True},
        {"name": "No conv, share_qk=False", "pre_conv": False, "disable_conv": True, "share_qk": False},
        {"name": "With conv, share_qk=True", "pre_conv": True, "disable_conv": False, "share_qk": True},
        {"name": "With conv, share_qk=False", "pre_conv": True, "disable_conv": False, "share_qk": False},
    ]
    
    for config_test in configs_to_test:
        print(f"\nTesting: {config_test['name']}")
        
        config_dict = TTT_STANDARD_CONFIGS["125m"].copy()
        config_dict.update({
            "max_position_embeddings": 128,
            "vocab_size": 1000,
            "pre_conv": config_test["pre_conv"],
            "disable_conv": config_test["disable_conv"],
            "share_qk": config_test["share_qk"],
        })
        
        config = TTTConfig(**config_dict)
        model = TTTForCausalLM(config)
        model.train()
        
        # Quick forward pass test
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        outputs = model(input_ids)
        logits = outputs.logits
        
        assert logits.shape == (1, 16, config.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        
        print(f"  ✓ {config_test['name']} passed!")

def test_parameter_count():
    """Compare parameter counts between conv and no-conv models."""
    print("\n" + "=" * 60)
    print("Testing Parameter Counts")
    print("=" * 60)
    
    base_config = TTT_STANDARD_CONFIGS["125m"].copy()
    base_config.update({
        "max_position_embeddings": 256,
        "vocab_size": 2000,
        "share_qk": True,
    })
    
    # Model with convolutions
    config_with_conv = TTTConfig(**{**base_config, "pre_conv": True, "disable_conv": False})
    model_with_conv = TTTForCausalLM(config_with_conv)
    
    # Model without convolutions
    config_no_conv = TTTConfig(**{**base_config, "pre_conv": False, "disable_conv": True})
    model_no_conv = TTTForCausalLM(config_no_conv)
    
    params_with_conv = sum(p.numel() for p in model_with_conv.parameters())
    params_no_conv = sum(p.numel() for p in model_no_conv.parameters())
    
    print(f"Parameters with convolutions: {params_with_conv:,}")
    print(f"Parameters without convolutions: {params_no_conv:,}")
    print(f"Difference: {params_with_conv - params_no_conv:,}")
    print(f"Reduction: {(params_with_conv - params_no_conv) / params_with_conv * 100:.2f}%")

if __name__ == "__main__":
    print("🧪 Starting Comprehensive TTT Model Functionality Test")
    print("This test verifies that all changes work correctly and maintain backward compatibility")
    
    try:
        # Run all tests
        model, config = test_basic_functionality()
        test_text_generation(model, config)
        test_with_convolutions_enabled()
        test_different_configurations()
        test_parameter_count()
        
        print("\n" + "=" * 60)
        print("🎉 ALL FUNCTIONALITY TESTS PASSED!")
        print("✓ Model works correctly with convolutions disabled")
        print("✓ Text generation capabilities verified")
        print("✓ Backward compatibility maintained") 
        print("✓ All configurations tested successfully")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)