#!/bin/bash
# Watchdog script for curriculum training - monitors and restarts on crash

set -e

CURRICULUM_DIR="curriculum_exp"
LOG_FILE="${CURRICULUM_DIR}/training.log"
WATCHDOG_LOG="${CURRICULUM_DIR}/watchdog.log"
MAX_RESTARTS=5

echo "[$(date)] 🐕 Watchdog started" | tee -a "${WATCHDOG_LOG}"

restart_count=0

while true; do
    # Check if training process is running
    if ! pgrep -f "train_curriculum_length_gen.py" > /dev/null; then
        
        # Check if we're done (30 checkpoints exist)
        checkpoint_count=$(find "${CURRICULUM_DIR}" -maxdepth 1 -type d -name "checkpoint_*" | wc -l)
        
        if [ "$checkpoint_count" -ge 30 ]; then
            echo "[$(date)] ✅ All 30 checkpoints complete! Watchdog stopping." | tee -a "${WATCHDOG_LOG}"
            break
        fi
        
        if [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
            echo "[$(date)] ❌ Max restarts ($MAX_RESTARTS) reached. Manual intervention needed." | tee -a "${WATCHDOG_LOG}"
            exit 1
        fi
        
        restart_count=$((restart_count + 1))
        echo "[$(date)] ⚠️  Training process not found. Restarting (attempt $restart_count/$MAX_RESTARTS)..." | tee -a "${WATCHDOG_LOG}"
        
        # Restart training
        cd /data/users/zacharie/ttt-lm-pytorch
        nohup python train_curriculum_length_gen.py \
            --output_dir curriculum_exp \
            --max_train_steps 15000 \
            --num_eval_checkpoints 30 \
            --eval_max_length 1280 \
            --per_device_train_batch_size 8 \
            --gradient_accumulation_steps 4 \
            --learning_rate 3e-4 \
            --mixed_precision bf16 \
            --seed 42 \
            --state_passing \
            --state_reset_interval 0 \
            --dataset_name wikitext \
            --dataset_config wikitext-103-raw-v1 \
            --dataset_subset_size -1 \
            >> "${LOG_FILE}" 2>&1 &
        
        echo "[$(date)] 🚀 Training restarted (PID: $!)" | tee -a "${WATCHDOG_LOG}"
        sleep 60  # Wait before checking again
    else
        # Training is running, check progress periodically
        if [ -f "${LOG_FILE}" ]; then
            last_line=$(tail -1 "${LOG_FILE}")
            if [[ "$last_line" == *"Step"* ]]; then
                echo "[$(date)] 💚 Training alive: $last_line" | tee -a "${WATCHDOG_LOG}"
            fi
        fi
        sleep 300  # Check every 5 minutes
    fi
done

echo "[$(date)] 🎉 Watchdog finished successfully" | tee -a "${WATCHDOG_LOG}"
