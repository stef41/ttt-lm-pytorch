#!/usr/bin/env python3
"""
Quick training test to check if model training is working efficiently.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS
import time

def test_training_speed():
    """Test training speed with current model."""
    print("Testing training speed...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model with current settings (convolutions disabled)
    config = TTTConfig(**{
        **TTT_STANDARD_CONFIGS["125m"],
        "pre_conv": False,
        "disable_conv": True,
        "max_position_embeddings": 128,
        "vocab_size": 1000,
    })
    
    print(f"Model config: pre_conv={config.pre_conv}, disable_conv={config.disable_conv}")
    
    model = TTTForCausalLM(config).to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    # Create fake dataset
    batch_size = 8
    seq_len = 64
    num_batches = 20
    
    # Generate fake data
    fake_data = []
    for _ in range(num_batches):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        fake_data.append(input_ids)
    
    print(f"Training on {num_batches} batches of size {batch_size}x{seq_len}")
    
    # Training loop
    total_loss = 0
    start_time = time.time()
    
    for step, input_ids in enumerate(fake_data):
        input_ids = input_ids.to(device)
        
        # Forward pass
        step_start = time.time()
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step_time = time.time() - step_start
        total_loss += loss.item()
        
        if step % 5 == 0:
            print(f"Step {step:2d}: loss={loss.item():.4f}, time={step_time:.3f}s, "
                  f"GPU mem={torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    total_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    avg_time_per_step = total_time / num_batches
    
    print(f"\n📊 TRAINING RESULTS:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Avg time per step: {avg_time_per_step:.3f}s")
    print(f"   Avg loss: {avg_loss:.4f}")
    print(f"   Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    print(f"   Steps per second: {num_batches/total_time:.2f}")

def test_model_loading_speed():
    """Test how long it takes to load the model."""
    print("\nTesting model loading speed...")
    
    start_time = time.time()
    
    config = TTTConfig(**{
        **TTT_STANDARD_CONFIGS["125m"],
        "pre_conv": False,
        "disable_conv": True,
    })
    
    config_time = time.time() - start_time
    
    model_start = time.time()
    model = TTTForCausalLM(config)
    model_time = time.time() - model_start
    
    gpu_start = time.time()
    if torch.cuda.is_available():
        model = model.to("cuda")
    gpu_time = time.time() - gpu_start
    
    print(f"✓ Config creation: {config_time:.3f}s")
    print(f"✓ Model creation: {model_time:.3f}s")
    print(f"✓ GPU transfer: {gpu_time:.3f}s")
    print(f"✓ Total loading time: {config_time + model_time + gpu_time:.3f}s")

def check_configuration():
    """Check if there are any configuration issues."""
    print("\nChecking model configuration...")
    
    config = TTTConfig(**{
        **TTT_STANDARD_CONFIGS["125m"],
        "pre_conv": False,
        "disable_conv": True,
    })
    
    print(f"✓ Model size: {config.hidden_size} hidden, {config.num_hidden_layers} layers")
    print(f"✓ Convolutions: pre_conv={config.pre_conv}, disable_conv={config.disable_conv}")
    print(f"✓ Share QK: {config.share_qk}")
    print(f"✓ TTT layer type: {config.ttt_layer_type}")
    print(f"✓ Mini batch size: {config.mini_batch_size}")
    
    # Check if config changes took effect
    model = TTTForCausalLM(config)
    
    # Check if conv layers were actually disabled
    has_conv_layers = False
    for name, module in model.named_modules():
        if 'conv' in name.lower() and hasattr(module, 'weight'):
            has_conv_layers = True
            print(f"Found conv layer: {name}")
    
    if not has_conv_layers and config.disable_conv:
        print("✓ Convolution layers successfully disabled!")
    elif has_conv_layers and not config.disable_conv:
        print("✓ Convolution layers present as expected")
    else:
        print("⚠️ Convolution layer state doesn't match config")

if __name__ == "__main__":
    print("="*60)
    print("TTT TRAINING SPEED TEST")
    print("="*60)
    
    try:
        check_configuration()
        test_model_loading_speed()
        test_training_speed()
        
        print("\n" + "="*60)
        print("✅ Training test completed successfully!")
        print("If training seems slow, it might be due to:")
        print("1. Large batch sizes or sequence lengths")
        print("2. Dataset loading/tokenization overhead")
        print("3. Logging/checkpointing frequency")
        print("4. Multi-GPU synchronization overhead")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)