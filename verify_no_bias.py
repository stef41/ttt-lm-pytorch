#!/usr/bin/env python3
"""
Verify that bias terms have been successfully removed from TTT weights.
"""

import torch
from ttt import TTTForCausalLM, TTT_STANDARD_CONFIGS, TTTConfig

def verify_no_bias():
    print("🔍 Verifying TTT model bias removal...")
    
    # Load config and create model
    config = TTT_STANDARD_CONFIGS["125m"].copy()
    config = TTTConfig(**config)
    
    model = TTTForCausalLM(config)
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Check TTT layers for bias parameters
    bias_found = False
    ttt_layers_checked = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'W1'):  # TTT layer
            ttt_layers_checked += 1
            print(f"\n📋 Checking TTT layer: {name}")
            
            # Check for W1
            if hasattr(module, 'W1'):
                print(f"  ✓ W1 shape: {module.W1.shape}")
            
            # Check for W2 (TTTMLP only)
            if hasattr(module, 'W2'):
                print(f"  ✓ W2 shape: {module.W2.shape}")
            
            # Check for bias terms (should not exist)
            if hasattr(module, 'b1'):
                print(f"  ❌ b1 found! Shape: {module.b1.shape}")
                bias_found = True
            else:
                print(f"  ✅ b1 not found (removed)")
                
            if hasattr(module, 'b2'):
                print(f"  ❌ b2 found! Shape: {module.b2.shape}")
                bias_found = True
            else:
                print(f"  ✅ b2 not found (removed)")
    
    print(f"\n📊 Summary:")
    print(f"  TTT layers checked: {ttt_layers_checked}")
    print(f"  Bias terms found: {'❌ YES' if bias_found else '✅ NO'}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if not bias_found:
        print("\n🎉 SUCCESS: All bias terms have been successfully removed from TTT weights!")
    else:
        print("\n❌ FAILURE: Some bias terms still exist in TTT layers!")
    
    return not bias_found

if __name__ == "__main__":
    verify_no_bias()