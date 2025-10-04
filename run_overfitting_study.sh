#!/bin/bash
# Launch overfitting study for both W=64 and W=128
# Runs full 100k steps without early stopping to observe overfitting

echo "=========================================="
echo "TTT OVERFITTING STUDY"
echo "=========================================="
echo "This will train TWO models for 100k steps each:"
echo "  1. W=64  (training window = 64 tokens)"
echo "  2. W=128 (training window = 128 tokens)"
echo ""
echo "NO EARLY STOPPING - Full training to study overfitting"
echo ""
echo "Estimated time per model: ~6-8 hours"
echo "Total estimated time: ~12-16 hours"
echo "=========================================="
echo ""

# Check if running both or just one
RUN_W64=true
RUN_W128=true

if [ "$1" == "w64" ]; then
    RUN_W128=false
    echo "Running W=64 only"
elif [ "$1" == "w128" ]; then
    RUN_W64=false
    echo "Running W=128 only"
else
    echo "Running both W=64 and W=128"
fi

echo ""

# Run W=64
if [ "$RUN_W64" = true ]; then
    echo "=========================================="
    echo "EXPERIMENT 1: W=64 (100k steps)"
    echo "=========================================="
    echo "Starting: $(date)"
    echo ""
    
    python train_overfitting_study.py \
        --max_seq_length 64 \
        --output_dir overfitting_w64 \
        --max_steps 100000 \
        --num_eval_checkpoints 50 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 3e-4 \
        2>&1 | tee overfitting_w64_training.log
    
    echo ""
    echo "W=64 completed: $(date)"
    echo ""
fi

# Run W=128
if [ "$RUN_W128" = true ]; then
    echo "=========================================="
    echo "EXPERIMENT 2: W=128 (100k steps)"
    echo "=========================================="
    echo "Starting: $(date)"
    echo ""
    
    python train_overfitting_study.py \
        --max_seq_length 128 \
        --output_dir overfitting_w128 \
        --max_steps 100000 \
        --num_eval_checkpoints 50 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 3e-4 \
        2>&1 | tee overfitting_w128_training.log
    
    echo ""
    echo "W=128 completed: $(date)"
    echo ""
fi

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Results saved in:"
if [ "$RUN_W64" = true ]; then
    echo "  - overfitting_w64/"
    echo "  - overfitting_w64_training.log"
fi
if [ "$RUN_W128" = true ]; then
    echo "  - overfitting_w128/"
    echo "  - overfitting_w128_training.log"
fi
echo ""
echo "To analyze results, run:"
echo "  python analyze_overfitting.py"
echo ""
