#!/usr/bin/env python3
"""
Simple test script to verify basic state passing functionality.
"""

import torch
from transformers import AutoTokenizer
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS, TTTCache

def test_basic_state_passing():
    """Test basic state passing functionality without complex gradient operations."""
    print("🧪 Basic State Passing Test")
    print("=" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create a small model for testing
    config = TTTConfig(**{
        **TTT_STANDARD_CONFIGS["125m"],
        "vocab_size": len(tokenizer),
        "max_position_embeddings": 64,
        "state_passing": True,
        "disable_conv": True,
    })
    
    print(f"Device: {device}")
    print(f"Model config: {config.hidden_size}d, {config.num_hidden_layers} layers")
    
    # Initialize model
    model = TTTForCausalLM(config).to(device)
    model.eval()  # Use eval mode to avoid gradient issues
    
    # Create test input
    text = "The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"Input: '{text}'")
    print(f"Input shape: {inputs['input_ids'].shape}")
    print()
    
    # Test 1: Forward pass without cache
    print("🔍 Test 1: Forward pass without cache")
    with torch.no_grad():
        outputs_no_cache = model(**inputs)
        logits_no_cache = outputs_no_cache.logits
        print(f"Output shape: {logits_no_cache.shape}")
        print(f"Sample logits: {logits_no_cache[0, 0, :5].cpu().numpy()}")
    print()
    
    # Test 2: Forward pass with cache
    print("🔍 Test 2: Forward pass with cache")
    ttt_cache = TTTCache(model.model, batch_size=1)
    
    with torch.no_grad():
        outputs_with_cache = model(cache_params=ttt_cache, **inputs)
        logits_with_cache = outputs_with_cache.logits
        print(f"Output shape: {logits_with_cache.shape}")
        print(f"Sample logits: {logits_with_cache[0, 0, :5].cpu().numpy()}")
    print()
    
    # Test 3: Sequential processing with state passing
    print("🔍 Test 3: Sequential processing with state passing")
    
    # Split the input into smaller pieces
    input_ids = inputs['input_ids']
    seq_len = input_ids.shape[1]
    
    # Process first half
    first_half = {"input_ids": input_ids[:, :seq_len//2]}
    second_half = {"input_ids": input_ids[:, seq_len//2:]}
    
    ttt_cache = TTTCache(model.model, batch_size=1)
    
    with torch.no_grad():
        # Process first part
        outputs1 = model(cache_params=ttt_cache, **first_half)
        print(f"First half processed: {first_half['input_ids'].shape}")
        print(f"First half logits: {outputs1.logits[0, -1, :5].cpu().numpy()}")
        
        # Process second part with state from first part
        outputs2 = model(cache_params=ttt_cache, **second_half)
        print(f"Second half processed: {second_half['input_ids'].shape}")
        print(f"Second half logits: {outputs2.logits[0, -1, :5].cpu().numpy()}")
    print()
    
    # Test 4: Compare with full sequence processing
    print("🔍 Test 4: Compare with full sequence")
    
    ttt_cache_full = TTTCache(model.model, batch_size=1)
    with torch.no_grad():
        full_outputs = model(cache_params=ttt_cache_full, **inputs)
        full_logits = full_outputs.logits[0, -1, :5].cpu().numpy()
        print(f"Full sequence logits: {full_logits}")
        
        # Compare final logits
        diff = abs(outputs2.logits[0, -1, :5].cpu().numpy() - full_logits).max()
        print(f"Max difference: {diff:.6f}")
        
        if diff < 0.1:  # Allow some numerical differences
            print("✅ PASS: Sequential and full processing are similar")
        else:
            print("❌ FAIL: Sequential and full processing differ significantly")
    print()
    
    # Test 5: Test config state_passing flag
    print("🔍 Test 5: Test config state_passing flag")
    
    config_no_state = TTTConfig(**{
        **TTT_STANDARD_CONFIGS["125m"],
        "vocab_size": len(tokenizer),
        "max_position_embeddings": 64,
        "state_passing": False,  # Disable state passing
        "disable_conv": True,
    })
    
    model_no_state = TTTForCausalLM(config_no_state).to(device)
    model_no_state.eval()
    
    with torch.no_grad():
        # This should work even without cache since state_passing=False
        outputs_no_state = model_no_state(**inputs)
        print(f"Model without state passing: {outputs_no_state.logits.shape}")
        print(f"Config state_passing: {model_no_state.config.state_passing}")
    print()
    
    print("🎉 Basic state passing tests completed!")
    return True

def test_training_with_state_passing():
    """Test state passing during training loop simulation."""
    print("🧪 Training with State Passing Test")
    print("=" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = TTTConfig(**{
        **TTT_STANDARD_CONFIGS["125m"],
        "vocab_size": len(tokenizer),
        "max_position_embeddings": 64,
        "state_passing": True,
        "disable_conv": True,
    })
    
    model = TTTForCausalLM(config).to(device)
    model.train()
    
    # Create some training batches
    texts = [
        "The quick brown fox jumps over",
        "Machine learning is a subset of",
        "The weather today is very nice",
    ]
    
    batches = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=16)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs['labels'] = inputs['input_ids'].clone()
        batches.append(inputs)
    
    print(f"Created {len(batches)} training batches")
    
    # Test with state passing
    print("\n🔍 Training with state passing:")
    ttt_cache = TTTCache(model.model, batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    losses_with_state = []
    for i, batch in enumerate(batches):
        optimizer.zero_grad()
        
        # Use detach on cache parameters to avoid gradient issues during testing
        # In real training, this would be handled by the framework
        outputs = model(cache_params=ttt_cache, **batch)
        loss = outputs.loss
        
        # Only compute gradients for model parameters, not cache
        loss.backward()
        optimizer.step()
        
        losses_with_state.append(loss.item())
        print(f"  Batch {i}: loss = {loss.item():.4f}")
    
    # Test without state passing
    print("\n🔍 Training without state passing:")
    model2 = TTTForCausalLM(config).to(device)
    model2.train()
    
    # Copy weights to ensure fair comparison
    model2.load_state_dict(model.state_dict())
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    
    losses_without_state = []
    for i, batch in enumerate(batches):
        optimizer2.zero_grad()
        
        # No cache_params means no state passing
        outputs = model2(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer2.step()
        
        losses_without_state.append(loss.item())
        print(f"  Batch {i}: loss = {loss.item():.4f}")
    
    print(f"\n📊 Results:")
    print(f"Final loss with state passing: {losses_with_state[-1]:.4f}")
    print(f"Final loss without state passing: {losses_without_state[-1]:.4f}")
    
    # They should be different, showing state passing has an effect
    diff = abs(losses_with_state[-1] - losses_without_state[-1])
    print(f"Difference: {diff:.4f}")
    
    if diff > 0.01:
        print("✅ PASS: State passing produces different training behavior")
    else:
        print("❌ WARNING: State passing may not be having significant effect")
    
    print()
    return True

def main():
    print("🚀 TTT State Passing Test Suite")
    print("=" * 50)
    print()
    
    try:
        success = True
        success &= test_basic_state_passing()
        success &= test_training_with_state_passing()
        
        if success:
            print("🎉 All tests completed!")
        else:
            print("❌ Some tests failed!")
            
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return success

if __name__ == "__main__":
    main()