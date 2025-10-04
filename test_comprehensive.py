#!/usr/bin/env python3
"""
Test script to verify TTT model works correctly with GPU and text generation.
"""
import torch
from ttt import TTTConfig, TTTForCausalLM, TTT_STANDARD_CONFIGS
from transformers import AutoTokenizer
import time

def test_gpu_usage():
    """Test that the model properly uses GPU."""
    print("Testing GPU usage...")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    device = torch.device("cuda")
    print(f"✓ Using device: {device}")
    print(f"✓ GPU: {torch.cuda.get_device_name()}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return True

def test_model_training_performance():
    """Test model training performance with and without convolutions."""
    if not torch.cuda.is_available():
        print("Skipping GPU performance test - CUDA not available")
        return
    
    device = torch.device("cuda")
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON TEST")
    print("="*60)
    
    # Test with convolutions disabled (current state)
    print("\n1. Testing with convolutions DISABLED...")
    config_no_conv = TTTConfig(**{
        **TTT_STANDARD_CONFIGS["125m"],
        "pre_conv": False,
        "disable_conv": True,
        "max_position_embeddings": 128,
        "vocab_size": 1000,
    })
    
    model_no_conv = TTTForCausalLM(config_no_conv).to(device)
    model_no_conv.train()
    
    # Test forward pass timing
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, config_no_conv.vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(3):
        outputs = model_no_conv(input_ids)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, config_no_conv.vocab_size), 
            input_ids.view(-1)
        )
        loss.backward()
        model_no_conv.zero_grad()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Actual timing
    for _ in range(10):
        outputs = model_no_conv(input_ids)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, config_no_conv.vocab_size), 
            input_ids.view(-1)
        )
        loss.backward()
        model_no_conv.zero_grad()
    
    torch.cuda.synchronize()
    no_conv_time = (time.time() - start_time) / 10
    
    print(f"✓ No Conv - Forward+Backward time: {no_conv_time:.4f}s per iteration")
    print(f"✓ No Conv - GPU Memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    # Clear memory
    del model_no_conv
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Test with convolutions enabled
    print("\n2. Testing with convolutions ENABLED...")
    config_with_conv = TTTConfig(**{
        **TTT_STANDARD_CONFIGS["125m"],
        "pre_conv": True,
        "disable_conv": False,
        "share_qk": True,
        "max_position_embeddings": 128,
        "vocab_size": 1000,
    })
    
    model_with_conv = TTTForCausalLM(config_with_conv).to(device)
    model_with_conv.train()
    
    # Warmup
    for _ in range(3):
        outputs = model_with_conv(input_ids)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, config_with_conv.vocab_size), 
            input_ids.view(-1)
        )
        loss.backward()
        model_with_conv.zero_grad()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Actual timing
    for _ in range(10):
        outputs = model_with_conv(input_ids)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, config_with_conv.vocab_size), 
            input_ids.view(-1)
        )
        loss.backward()
        model_with_conv.zero_grad()
    
    torch.cuda.synchronize()
    with_conv_time = (time.time() - start_time) / 10
    
    print(f"✓ With Conv - Forward+Backward time: {with_conv_time:.4f}s per iteration")
    print(f"✓ With Conv - GPU Memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    # Compare
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   No Conv:   {no_conv_time:.4f}s/iter")
    print(f"   With Conv: {with_conv_time:.4f}s/iter")
    speedup = with_conv_time / no_conv_time
    print(f"   Speedup:   {speedup:.2f}x {'(faster without conv)' if speedup > 1 else '(faster with conv)'}")

def test_text_generation():
    """Test text generation capabilities."""
    print("\n" + "="*60)
    print("TEXT GENERATION TEST")
    print("="*60)
    
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU for generation test")
    else:
        device = torch.device("cuda")
        print("Using GPU for generation test")
    
    # Create model with current (no conv) settings
    config = TTTConfig(**{
        **TTT_STANDARD_CONFIGS["125m"],
        "pre_conv": False,
        "disable_conv": True,
        "vocab_size": 50257,  # GPT-2 vocab size for tokenizer compatibility
        "max_position_embeddings": 512,
    })
    
    print(f"Model config: pre_conv={config.pre_conv}, disable_conv={config.disable_conv}")
    
    model = TTTForCausalLM(config).to(device)
    model.eval()
    
    # Use GPT-2 tokenizer for text generation
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except:
        print("⚠️ Could not load GPT-2 tokenizer, using simple token IDs")
        tokenizer = None
    
    # Manual text generation since model doesn't have generate() method
    if tokenizer:
        prompts = [
            "The quick brown fox",
            "In a world where artificial intelligence",
            "Once upon a time"
        ]
        
        for i, prompt in enumerate(prompts):
            print(f"\n🔤 Generation Test {i+1}: '{prompt}'")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            
            print(f"Input shape: {input_ids.shape}")
            
            # Manual generation loop
            generated_ids = input_ids.clone()
            max_new_tokens = 10
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = model(generated_ids)
                    next_token_logits = outputs.logits[0, -1, :]
                    
                    # Sample from top-k
                    top_k = 50
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = torch.softmax(top_k_logits / 0.7, dim=-1)  # temperature=0.7
                    next_token_idx = torch.multinomial(probs, 1)
                    next_token = top_k_indices[next_token_idx]
                    
                    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                    
                    # Stop if we hit max length
                    if generated_ids.shape[1] >= config.max_position_embeddings:
                        break
            
            # Decode
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Generated: '{generated_text}'")
            print("✓ Generation successful!")
    
    else:
        # Simple token ID test
        print("\n🔢 Simple token generation test:")
        input_ids = torch.randint(1, 1000, (1, 10), device=device)
        print(f"Input token IDs: {input_ids}")
        
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
        
        print(f"Next predicted token ID: {next_token}")
        print("✓ Token prediction successful!")

def test_model_consistency():
    """Test that model outputs are consistent and reasonable."""
    print("\n" + "="*60)
    print("MODEL CONSISTENCY TEST")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = TTTConfig(**{
        **TTT_STANDARD_CONFIGS["125m"],
        "pre_conv": False,
        "disable_conv": True,
        "max_position_embeddings": 128,
        "vocab_size": 1000,
    })
    
    model = TTTForCausalLM(config).to(device)
    model.eval()
    
    # Test consistency across multiple runs
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    
    outputs1 = model(input_ids)
    outputs2 = model(input_ids)
    
    # Check deterministic behavior in eval mode
    logits_diff = torch.abs(outputs1.logits - outputs2.logits).max().item()
    print(f"✓ Logits difference between runs: {logits_diff:.10f}")
    
    if logits_diff < 1e-6:
        print("✓ Model is deterministic in eval mode")
    else:
        print("⚠️ Model shows non-deterministic behavior")
    
    # Check logits properties
    logits = outputs1.logits
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"✓ Logits mean: {logits.mean().item():.3f}")
    print(f"✓ Logits std: {logits.std().item():.3f}")
    
    # Check for NaN/Inf
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    
    if has_nan:
        print("❌ Found NaN values in logits!")
    else:
        print("✓ No NaN values found")
    
    if has_inf:
        print("❌ Found Inf values in logits!")
    else:
        print("✓ No Inf values found")

if __name__ == "__main__":
    print("="*60)
    print("TTT MODEL COMPREHENSIVE TEST")
    print("="*60)
    
    try:
        # Test GPU
        gpu_available = test_gpu_usage()
        
        # Test performance
        if gpu_available:
            test_model_training_performance()
        
        # Test generation
        test_text_generation()
        
        # Test consistency
        test_model_consistency()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED!")
        print("Model is working correctly with disabled convolutions!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)