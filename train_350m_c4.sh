#!/bin/bash
# Training script for TTT-350M on C4 dataset
# Based on official JAX training recipe
# Adapted for PyTorch implementation

# Dataset configuration
DATASET_NAME="allenai/c4"
DATASET_CONFIG="en"

# Training hyperparameters (from JAX recipe)
# Product should equal 0.5 million tokens
SEQ_LEN=2048
BATCH_SIZE=256  # Global batch size = 256 sequences = 0.5M tokens per step
TOTAL_STEPS=13500

# Calculate per-device batch size for distributed training
# Assuming 8 GPUs, each will process 32 sequences
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
PER_DEVICE_BS=$((BATCH_SIZE / NUM_GPUS))
GRAD_ACCUM_STEPS=1

# If batch doesn't divide evenly, use gradient accumulation
if [ $((BATCH_SIZE % NUM_GPUS)) -ne 0 ]; then
    PER_DEVICE_BS=16
    GRAD_ACCUM_STEPS=$((BATCH_SIZE / (NUM_GPUS * PER_DEVICE_BS)))
fi

# Experiment details
EXP_NAME="ttt_350m_c4_$(date +%Y%m%d_%H%M%S)"
EXP_DIR="./experiments"

# Create experiment directory
mkdir -p ${EXP_DIR}/${EXP_NAME}
echo "Starting training for TTT-350M"
echo "Experiment: ${EXP_NAME}"
echo "Global batch size: ${BATCH_SIZE} sequences (${BATCH_SIZE}*${SEQ_LEN} = $((BATCH_SIZE*SEQ_LEN/1000))K tokens/step)"
echo "Per-device batch size: ${PER_DEVICE_BS}"
echo "Gradient accumulation steps: ${GRAD_ACCUM_STEPS}"
echo "Number of GPUs: ${NUM_GPUS}"
echo ""

# Training command
accelerate launch --config_file accelerate_config.yaml \
    train_ttt.py \
    --model_size "350m" \
    --ttt_layer_type "linear" \
    --dataset_name ${DATASET_NAME} \
    --dataset_config ${DATASET_CONFIG} \
    --seq_length ${SEQ_LEN} \
    --per_device_train_batch_size ${PER_DEVICE_BS} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --max_train_steps ${TOTAL_STEPS} \
    --learning_rate 1.5e-3 \
    --lr_end 1e-5 \
    --lr_warmup_steps 1350 \
    --lr_decay_steps ${TOTAL_STEPS} \
    --weight_decay 0.1 \
    --optimizer "adamw" \
    --mixed_precision "bf16" \
    --state_passing \
    --ttt_base_lr 1.0 \
    --save_checkpoint_freq 1000 \
    --save_milestone_freq 2000 \
    --output_dir ${EXP_DIR}/${EXP_NAME} \
    --logging_steps 10 \
    --seed 42 \
    --wandb_project "ttt-lm-pytorch" \
    --wandb_run_name ${EXP_NAME}

echo ""
echo "Training complete! Model saved to: ${EXP_DIR}/${EXP_NAME}"
