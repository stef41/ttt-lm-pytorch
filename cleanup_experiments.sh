#!/bin/bash
# Cleanup script to remove experiment directories
# WARNING: This will delete ~4TB of experiment data!
# Generated on 2025-10-16

echo "WARNING: This script will remove large experiment directories (~4TB total)."
echo "The following directories will be removed:"
echo ""
echo "  - length_gen_w128_with_ckpts (1.3TB)"
echo "  - length_gen_w32_with_ckpts (1.3TB)"
echo "  - length_gen_w64_with_ckpts (1.3TB)"
echo "  - archive_invalid_no_state_passing (1.5GB)"
echo "  - archived_experiments (39MB)"
echo "  - length_gen_study_w128 (514MB)"
echo "  - length_gen_study_wt2 (509MB)"
echo "  - length_gen_study (509MB)"
echo "  - length_gen_study_w32 (502MB)"
echo "  - length_gen_study_w64_500ckpt (274MB)"
echo "  - length_gen_study_w64_500ckpt_no_sp (255MB)"
echo "  - length_gen_study_w32_500ckpt (235MB)"
echo "  - length_gen_study_w32_500ckpt_no_sp (234MB)"
echo "  - length_gen_study_w16_500ckpt (197MB)"
echo "  - length_gen_study_w16_500ckpt_no_sp (197MB)"
echo "  - archive_invalid_experiments (4KB)"
echo "  - long_training_wt103 (12KB)"
echo "  - curriculum_exp_long (8KB)"
echo ""
read -p "Are you sure you want to proceed? (type 'yes' to confirm): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Starting cleanup of experiment directories..."

# Remove large checkpoint directories
echo "Removing large checkpoint directories (this may take a while)..."
rm -rf length_gen_w128_with_ckpts
rm -rf length_gen_w32_with_ckpts
rm -rf length_gen_w64_with_ckpts

# Remove archived/invalid experiments
echo "Removing archived and invalid experiments..."
rm -rf archive_invalid_no_state_passing
rm -rf archived_experiments
rm -rf archive_invalid_experiments

# Remove study directories
echo "Removing study directories..."
rm -rf length_gen_study_w128
rm -rf length_gen_study_wt2
rm -rf length_gen_study
rm -rf length_gen_study_w32
rm -rf length_gen_study_w64_500ckpt
rm -rf length_gen_study_w64_500ckpt_no_sp
rm -rf length_gen_study_w32_500ckpt
rm -rf length_gen_study_w32_500ckpt_no_sp
rm -rf length_gen_study_w16_500ckpt
rm -rf length_gen_study_w16_500ckpt_no_sp

# Remove other experiment directories
echo "Removing other experiment directories..."
rm -rf long_training_wt103
rm -rf curriculum_exp_long

# Clean up empty log directories
echo "Cleaning up empty directories..."
rm -rf logs
rm -rf 100xW_evaluations

echo ""
echo "Cleanup complete!"
echo "Approximately 4TB of space has been freed."
