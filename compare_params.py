#!/usr/bin/env python3
"""
Compare parameter counts before/after bias removal.
"""

# Previous parameter count (with bias): 124,298,064
# Current parameter count (no bias): 124,288,848

previous_params = 124_298_064
current_params = 124_288_848
reduction = previous_params - current_params

print("📊 TTT Parameter Count Comparison:")
print("="*50)
print(f"With bias terms:    {previous_params:,} parameters")
print(f"Without bias terms: {current_params:,} parameters")
print(f"Reduction:          {reduction:,} parameters")
print(f"Reduction %:        {(reduction/previous_params)*100:.3f}%")

# Calculate expected reduction for TTT-Linear
# 12 layers × 12 heads × (1 × 64) bias terms = 9,216 parameters
expected_reduction = 12 * 12 * 64  # layers × heads × head_dim (b1)
print(f"\nExpected reduction: {expected_reduction:,} parameters")
print(f"Actual matches expected: {'✅ YES' if reduction == expected_reduction else '❌ NO'}")

print("\n🎉 Bias removal completed successfully!")