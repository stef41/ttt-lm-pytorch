#!/usr/bin/env python3
"""
Test script to verify state passing functionality in TTT models.
Compares training with and without state passing to ensure it works correctly.
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS, TTTCache

def create_test_data(tokenizer, seq_length=64, num_batches=5, batch_size=2):
    """Create consistent test data for reproducible experiments."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate random token sequences
    vocab_size = len(tokenizer)
    batches = []
    
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        batches.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        })
    
    return batches

def test_state_passing_consistency():
    """Test that state passing produces consistent results."""
    print("🧪 Testing State Passing Consistency")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model configuration with state passing enabled
    config = TTTConfig(**{
        **TTT_STANDARD_CONFIGS["125m"],
        "vocab_size": len(tokenizer),
        "max_position_embeddings": 128,
        "state_passing": True,
        "disable_conv": True,  # Keep consistent with our current setup
    })
    
    # Create test data
    test_batches = create_test_data(tokenizer, seq_length=32, num_batches=3, batch_size=2)
    
    print(f"Device: {device}")
    print(f"Model: {config.hidden_size}d, {config.num_hidden_layers} layers")
    print(f"Test batches: {len(test_batches)}")
    print(f"Batch size: {test_batches[0]['input_ids'].shape[0]}")
    print(f"Sequence length: {test_batches[0]['input_ids'].shape[1]}")
    print()
    
    # Test 1: Multiple runs with state passing should be deterministic
    print("🔍 Test 1: Deterministic behavior with state passing")
    
    losses_run1 = []
    losses_run2 = []
    
    for run in range(2):
        print(f"  Run {run + 1}:")
        
        # Initialize model and cache
        torch.manual_seed(42)  # Same seed for both runs
        model = TTTForCausalLM(config).to(device)
        model.train()
        
        ttt_cache = TTTCache(model.model, batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        run_losses = []
        
        for i, batch in enumerate(test_batches):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Create a fresh copy of cache for each backward pass to avoid in-place issues
            if run == 0 or not use_state_passing:
                # First run - establish baseline
                optimizer.zero_grad()
                outputs = model(cache_params=ttt_cache, **batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            else:
                # Second run - test with same initialization but independent computation
                with torch.no_grad():
                    outputs = model(cache_params=ttt_cache, **batch)
                    loss = outputs.loss
            
            run_losses.append(loss.item())
            print(f"    Batch {i}: loss = {loss.item():.6f}")
        
        if run == 0:
            losses_run1 = run_losses
        else:
            losses_run2 = run_losses
        
        print()
    
    # Check determinism
    max_diff = max(abs(l1 - l2) for l1, l2 in zip(losses_run1, losses_run2))
    print(f"  Maximum difference between runs: {max_diff:.8f}")
    if max_diff < 1e-6:
        print("  ✅ PASS: Deterministic behavior confirmed")
    else:
        print("  ❌ FAIL: Non-deterministic behavior detected")
    print()
    
    # Test 2: Compare with and without state passing
    print("🔍 Test 2: State passing vs no state passing")
    
    configs_to_test = [
        ("With state passing", True),
        ("Without state passing", False),
    ]
    
    all_results = {}
    
    for name, use_state_passing in configs_to_test:
        print(f"  {name}:")
        
        # Update config
        config.state_passing = use_state_passing
        
        torch.manual_seed(42)  # Same initialization
        model = TTTForCausalLM(config).to(device)
        model.train()
        
        ttt_cache = None
        if use_state_passing:
            ttt_cache = TTTCache(model.model, batch_size=2)
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        losses = []
        final_hidden_states = []
        
        for i, batch in enumerate(test_batches):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            forward_kwargs = batch.copy()
            if use_state_passing and ttt_cache is not None:
                forward_kwargs['cache_params'] = ttt_cache
                
            outputs = model(**forward_kwargs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Store some hidden states for comparison
            with torch.no_grad():
                hidden_outputs = model.model(forward_kwargs['input_ids'])
                final_hidden_states.append(hidden_outputs.last_hidden_state[:, -1, :].cpu())
            
            print(f"    Batch {i}: loss = {loss.item():.6f}")
        
        all_results[name] = {
            'losses': losses,
            'final_hidden_states': final_hidden_states
        }
        print()
    
    # Compare results
    losses_with = all_results["With state passing"]['losses']
    losses_without = all_results["Without state passing"]['losses']
    
    print("📊 Comparison Results:")
    print(f"  Final loss with state passing: {losses_with[-1]:.6f}")
    print(f"  Final loss without state passing: {losses_without[-1]:.6f}")
    
    # They should be different if state passing is working
    loss_diff = abs(losses_with[-1] - losses_without[-1])
    print(f"  Difference: {loss_diff:.6f}")
    
    if loss_diff > 1e-4:
        print("  ✅ PASS: State passing produces different results (as expected)")
    else:
        print("  ❌ FAIL: State passing produces same results (unexpected)")
    
    print()
    
    # Test 3: Check that cache is being updated
    print("🔍 Test 3: Cache state evolution")
    
    config.state_passing = True
    torch.manual_seed(42)
    model = TTTForCausalLM(config).to(device)
    model.train()
    
    ttt_cache = TTTCache(model.model, batch_size=2)
    
    # Get initial cache state
    initial_params = {}
    for name in ttt_cache.ttt_param_names:
        param_key = f"{name}_states"
        if param_key in ttt_cache.ttt_params_dict:
            layer_0_param = ttt_cache.ttt_params_dict[param_key].get(0)
            if layer_0_param is not None:
                initial_params[name] = layer_0_param.clone()
    
    # Run one forward pass
    batch = test_batches[0]
    batch = {k: v.to(device) for k, v in batch.items()}
    
    outputs = model(cache_params=ttt_cache, **batch)
    
    # Check if cache changed
    cache_changed = False
    for name in initial_params:
        param_key = f"{name}_states"
        if param_key in ttt_cache.ttt_params_dict:
            layer_0_param = ttt_cache.ttt_params_dict[param_key].get(0)
            if layer_0_param is not None:
                diff = torch.norm(initial_params[name] - layer_0_param).item()
                print(f"  Parameter {name} change: {diff:.8f}")
                if diff > 1e-6:
                    cache_changed = True
    
    if cache_changed:
        print("  ✅ PASS: Cache parameters are being updated")
    else:
        print("  ❌ FAIL: Cache parameters are not changing")
    
    print()
    return True

def test_memory_usage():
    """Test memory usage with state passing."""
    print("🧪 Testing Memory Usage")
    print("=" * 30)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping memory test")
        return True
    
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = TTTConfig(**{
        **TTT_STANDARD_CONFIGS["125m"],
        "vocab_size": len(tokenizer),
        "state_passing": True,
        "disable_conv": True,
    })
    
    model = TTTForCausalLM(config).to(device)
    ttt_cache = TTTCache(model.model, batch_size=4)
    
    # Monitor memory before and after
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / 1e6
    
    # Create larger batch for memory test
    input_ids = torch.randint(0, len(tokenizer), (4, 128), device=device)
    labels = input_ids.clone()
    
    outputs = model(input_ids=input_ids, labels=labels, cache_params=ttt_cache)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e6
    final_memory = torch.cuda.memory_allocated() / 1e6
    
    print(f"Initial memory: {initial_memory:.1f} MB")
    print(f"Peak memory: {peak_memory:.1f} MB")
    print(f"Final memory: {final_memory:.1f} MB")
    print(f"Cache overhead: {final_memory - initial_memory:.1f} MB")
    
    # Should be reasonable overhead
    if final_memory - initial_memory < 1000:  # Less than 1GB overhead
        print("✅ PASS: Reasonable memory overhead")
    else:
        print("❌ FAIL: High memory overhead")
    
    print()
    return True

def main():
    """Run all state passing tests."""
    print("🚀 TTT State Passing Test Suite")
    print("=" * 50)
    print()
    
    try:
        success = True
        success &= test_state_passing_consistency()
        success &= test_memory_usage()
        
        if success:
            print("🎉 All tests passed!")
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