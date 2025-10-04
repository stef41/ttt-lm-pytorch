#!/usr/bin/env python3
"""
Find the maximum batch size supported by the TTT model with disabled convolutions.
"""
import torch
import gc
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS

def test_batch_size(batch_size, seq_length=128, model_size="125m"):
    """Test if a specific batch size fits in GPU memory."""
    try:
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        device = torch.device("cuda")
        
        # Create model with disabled convolutions
        config = TTTConfig(**{
            **TTT_STANDARD_CONFIGS[model_size],
            "pre_conv": False,
            "disable_conv": True,
            "max_position_embeddings": seq_length * 2,  # Give some buffer
            "vocab_size": 50257,  # GPT-2 vocab size
        })
        
        model = TTTForCausalLM(config).to(device)
        model.train()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        
        # Test input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
        
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1)
        )
        
        # Backward pass (this is where memory usage peaks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        # Clean up
        del model, optimizer, input_ids, outputs, logits, loss
        torch.cuda.empty_cache()
        gc.collect()
        
        return True, allocated, reserved, max_allocated
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clean up on OOM
            torch.cuda.empty_cache()
            gc.collect()
            return False, 0, 0, 0
        else:
            raise e

def find_max_batch_size(seq_length=128, model_size="125m"):
    """Binary search to find maximum batch size."""
    print(f"Finding maximum batch size for {model_size} model with {seq_length} token sequences...")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Start with a reasonable range
    min_batch = 1
    max_batch = 512  # Start high, will reduce if needed
    
    # First, find an upper bound that fails
    print("🔍 Finding upper bound...")
    test_batch = max_batch
    while test_batch >= min_batch:
        success, alloc, reserved, max_alloc = test_batch_size(test_batch, seq_length, model_size)
        if success:
            print(f"✅ Batch size {test_batch}: SUCCESS (peak: {max_alloc:.2f}GB)")
            min_batch = test_batch
            break
        else:
            print(f"❌ Batch size {test_batch}: OOM")
            max_batch = test_batch - 1
            test_batch = test_batch // 2
    
    if test_batch < min_batch:
        # Even smallest batch failed
        print("❌ Even batch size 1 failed!")
        return 0
    
    # Binary search for the exact maximum
    print(f"\n🎯 Binary search between {min_batch} and {max_batch}...")
    
    last_successful = min_batch
    last_successful_stats = None
    
    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2
        success, alloc, reserved, max_alloc = test_batch_size(mid_batch, seq_length, model_size)
        
        if success:
            print(f"✅ Batch size {mid_batch}: SUCCESS (peak: {max_alloc:.2f}GB)")
            last_successful = mid_batch
            last_successful_stats = (alloc, reserved, max_alloc)
            min_batch = mid_batch + 1
        else:
            print(f"❌ Batch size {mid_batch}: OOM")
            max_batch = mid_batch - 1
    
    return last_successful, last_successful_stats

def test_multiple_configurations():
    """Test different model sizes and sequence lengths."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    configurations = [
        ("125m", 64),
        ("125m", 128),
        ("125m", 256),
        ("125m", 512),
        ("350m", 64),
        ("350m", 128),
        ("350m", 256),
        ("760m", 64),
        ("760m", 128),
        ("1b", 64),
        ("1b", 128),
    ]
    
    results = []
    
    print("="*80)
    print("TTT MODEL MAXIMUM BATCH SIZE FINDER")
    print("="*80)
    
    for model_size, seq_length in configurations:
        print(f"\n{'='*60}")
        print(f"Testing {model_size} model with {seq_length} tokens")
        print(f"{'='*60}")
        
        try:
            max_batch, stats = find_max_batch_size(seq_length, model_size)
            
            if max_batch > 0:
                alloc, reserved, max_alloc = stats
                total_tokens = max_batch * seq_length
                
                print(f"\n🎉 RESULT: Maximum batch size = {max_batch}")
                print(f"   Total tokens per batch: {total_tokens:,}")
                print(f"   Peak GPU memory: {max_alloc:.2f}GB")
                print(f"   Memory efficiency: {total_tokens/max_alloc/1000:.0f}K tokens/GB")
                
                results.append({
                    'model': model_size,
                    'seq_len': seq_length,
                    'max_batch': max_batch,
                    'total_tokens': total_tokens,
                    'peak_memory': max_alloc,
                    'efficiency': total_tokens/max_alloc/1000
                })
            else:
                print(f"\n❌ FAILED: Could not fit even batch size 1")
                results.append({
                    'model': model_size,
                    'seq_len': seq_length,
                    'max_batch': 0,
                    'total_tokens': 0,
                    'peak_memory': 0,
                    'efficiency': 0
                })
                
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            results.append({
                'model': model_size,
                'seq_len': seq_length,
                'max_batch': -1,
                'total_tokens': -1,
                'peak_memory': -1,
                'efficiency': -1
            })
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Model':<8} {'SeqLen':<8} {'MaxBatch':<10} {'TotalTokens':<12} {'PeakMem':<10} {'Efficiency':<12}")
    print(f"{'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*12}")
    
    for r in results:
        if r['max_batch'] > 0:
            print(f"{r['model']:<8} {r['seq_len']:<8} {r['max_batch']:<10} {r['total_tokens']:<12,} {r['peak_memory']:<10.1f}GB {r['efficiency']:<12.0f}K/GB")
        elif r['max_batch'] == 0:
            print(f"{r['model']:<8} {r['seq_len']:<8} {'FAIL':<10} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
        else:
            print(f"{r['model']:<8} {r['seq_len']:<8} {'ERROR':<10} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
    
    # Find best configurations
    valid_results = [r for r in results if r['max_batch'] > 0]
    if valid_results:
        best_tokens = max(valid_results, key=lambda x: x['total_tokens'])
        best_efficiency = max(valid_results, key=lambda x: x['efficiency'])
        
        print(f"\n🏆 BEST CONFIGURATIONS:")
        print(f"   Highest throughput: {best_tokens['model']} + {best_tokens['seq_len']} tokens = {best_tokens['total_tokens']:,} tokens/batch")
        print(f"   Best efficiency: {best_efficiency['model']} + {best_efficiency['seq_len']} tokens = {best_efficiency['efficiency']:.0f}K tokens/GB")

if __name__ == "__main__":
    test_multiple_configurations()