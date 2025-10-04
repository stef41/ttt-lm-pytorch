#!/bin/bash
# Continuous monitoring for overfitting study with crash detection

echo "=========================================="
echo "TTT OVERFITTING STUDY - CRASH MONITOR"
echo "=========================================="
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checking every 60 seconds for crashes..."
echo "Press Ctrl+C to stop monitoring"
echo ""

# Initialize state tracking
PREV_W64_RUNNING=false
PREV_W128_RUNNING=false
W64_CRASH_COUNT=0
W128_CRASH_COUNT=0

while true; do
    clear
    echo "=========================================="
    echo "CRASH MONITORING - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Check W=64
    W64_RUNNING=false
    if pgrep -f "train_overfitting_study.py.*overfitting_w64" > /dev/null; then
        W64_RUNNING=true
    fi
    
    # Check W=128
    W128_RUNNING=false
    if pgrep -f "train_overfitting_study.py.*overfitting_w128" > /dev/null; then
        W128_RUNNING=true
    fi
    
    # W=64 Status and crash detection
    echo "W=64 Experiment:"
    echo "----------------"
    if [ "$W64_RUNNING" = true ]; then
        PID=$(pgrep -f "train_overfitting_study.py.*overfitting_w64")
        echo "✓ RUNNING (PID: $PID)"
        
        # Check for errors in recent log
        if [ -f "overfitting_w64_training.log" ]; then
            ERROR_COUNT=$(tail -100 overfitting_w64_training.log | grep -i -E "(error|exception|traceback|failed|killed)" | wc -l)
            if [ $ERROR_COUNT -gt 0 ]; then
                echo "⚠️  WARNING: $ERROR_COUNT error messages in recent log!"
                echo ""
                echo "Recent errors:"
                tail -100 overfitting_w64_training.log | grep -i -E "(error|exception|traceback|failed)" | tail -5
            fi
            
            # Show progress
            RECENT_PROGRESS=$(tail -20 overfitting_w64_training.log | grep "Training:" | tail -1)
            if [ ! -z "$RECENT_PROGRESS" ]; then
                echo ""
                echo "Progress: $RECENT_PROGRESS"
            fi
            
            # Show latest step
            LATEST_STEP=$(tail -50 overfitting_w64_training.log | grep -oP "Step \K[0-9]+" | tail -1)
            if [ ! -z "$LATEST_STEP" ]; then
                echo "Latest step: $LATEST_STEP / 100,000"
                PERCENT=$((LATEST_STEP * 100 / 100000))
                echo "Completion: ${PERCENT}%"
            fi
        fi
        
        # Check checkpoints
        if [ -d "overfitting_w64" ]; then
            CKPT_COUNT=$(ls -1d overfitting_w64/checkpoint_* 2>/dev/null | wc -l)
            echo "Checkpoints: $CKPT_COUNT / 50"
        fi
        
        W64_CRASH_COUNT=0
    elif [ "$PREV_W64_RUNNING" = true ] && [ "$W64_RUNNING" = false ]; then
        # Process was running, now stopped - check if crash or completion
        W64_CRASH_COUNT=$((W64_CRASH_COUNT + 1))
        
        if [ -f "overfitting_w64_training.log" ]; then
            # Check last 50 lines for completion or error
            COMPLETED=$(tail -50 overfitting_w64_training.log | grep -i "TRAINING COMPLETE" | wc -l)
            ERROR_EXIT=$(tail -50 overfitting_w64_training.log | grep -i -E "(error|exception|traceback|killed|terminated)" | wc -l)
            
            if [ $COMPLETED -gt 0 ]; then
                echo "✓ COMPLETED SUCCESSFULLY"
                W64_CRASH_COUNT=0
            elif [ $ERROR_EXIT -gt 0 ]; then
                echo "❌ CRASHED! (Crash count: $W64_CRASH_COUNT)"
                echo ""
                echo "Error details (last 30 lines):"
                tail -30 overfitting_w64_training.log | grep -A 10 -B 5 -i -E "(error|exception|traceback)"
                echo ""
                echo "Full error context saved to: overfitting_w64_crash.log"
                tail -100 overfitting_w64_training.log > overfitting_w64_crash.log
            else
                echo "⚠️  STOPPED UNEXPECTEDLY (Possible crash)"
                echo ""
                echo "Last 20 lines of log:"
                tail -20 overfitting_w64_training.log
            fi
        else
            echo "❌ PROCESS STOPPED (no log file found)"
        fi
    elif [ -d "overfitting_w64" ]; then
        CKPT_COUNT=$(ls -1d overfitting_w64/checkpoint_* 2>/dev/null | wc -l)
        if [ $CKPT_COUNT -ge 50 ]; then
            echo "✓ COMPLETED ($CKPT_COUNT checkpoints)"
        else
            echo "⏸️  IDLE ($CKPT_COUNT checkpoints exist)"
        fi
    else
        echo "⏳ NOT STARTED"
    fi
    
    echo ""
    
    # W=128 Status and crash detection
    echo "W=128 Experiment:"
    echo "-----------------"
    if [ "$W128_RUNNING" = true ]; then
        PID=$(pgrep -f "train_overfitting_study.py.*overfitting_w128")
        echo "✓ RUNNING (PID: $PID)"
        
        # Check for errors in recent log
        if [ -f "overfitting_w128_training.log" ]; then
            ERROR_COUNT=$(tail -100 overfitting_w128_training.log | grep -i -E "(error|exception|traceback|failed|killed)" | wc -l)
            if [ $ERROR_COUNT -gt 0 ]; then
                echo "⚠️  WARNING: $ERROR_COUNT error messages in recent log!"
                echo ""
                echo "Recent errors:"
                tail -100 overfitting_w128_training.log | grep -i -E "(error|exception|traceback|failed)" | tail -5
            fi
            
            # Show progress
            RECENT_PROGRESS=$(tail -20 overfitting_w128_training.log | grep "Training:" | tail -1)
            if [ ! -z "$RECENT_PROGRESS" ]; then
                echo ""
                echo "Progress: $RECENT_PROGRESS"
            fi
            
            # Show latest step
            LATEST_STEP=$(tail -50 overfitting_w128_training.log | grep -oP "Step \K[0-9]+" | tail -1)
            if [ ! -z "$LATEST_STEP" ]; then
                echo "Latest step: $LATEST_STEP / 100,000"
                PERCENT=$((LATEST_STEP * 100 / 100000))
                echo "Completion: ${PERCENT}%"
            fi
        fi
        
        # Check checkpoints
        if [ -d "overfitting_w128" ]; then
            CKPT_COUNT=$(ls -1d overfitting_w128/checkpoint_* 2>/dev/null | wc -l)
            echo "Checkpoints: $CKPT_COUNT / 50"
        fi
        
        W128_CRASH_COUNT=0
    elif [ "$PREV_W128_RUNNING" = true ] && [ "$W128_RUNNING" = false ]; then
        # Process was running, now stopped - check if crash or completion
        W128_CRASH_COUNT=$((W128_CRASH_COUNT + 1))
        
        if [ -f "overfitting_w128_training.log" ]; then
            # Check last 50 lines for completion or error
            COMPLETED=$(tail -50 overfitting_w128_training.log | grep -i "TRAINING COMPLETE" | wc -l)
            ERROR_EXIT=$(tail -50 overfitting_w128_training.log | grep -i -E "(error|exception|traceback|killed|terminated)" | wc -l)
            
            if [ $COMPLETED -gt 0 ]; then
                echo "✓ COMPLETED SUCCESSFULLY"
                W128_CRASH_COUNT=0
            elif [ $ERROR_EXIT -gt 0 ]; then
                echo "❌ CRASHED! (Crash count: $W128_CRASH_COUNT)"
                echo ""
                echo "Error details (last 30 lines):"
                tail -30 overfitting_w128_training.log | grep -A 10 -B 5 -i -E "(error|exception|traceback)"
                echo ""
                echo "Full error context saved to: overfitting_w128_crash.log"
                tail -100 overfitting_w128_training.log > overfitting_w128_crash.log
            else
                echo "⚠️  STOPPED UNEXPECTEDLY (Possible crash)"
                echo ""
                echo "Last 20 lines of log:"
                tail -20 overfitting_w128_training.log
            fi
        else
            echo "❌ PROCESS STOPPED (no log file found)"
        fi
    elif [ -d "overfitting_w128" ]; then
        CKPT_COUNT=$(ls -1d overfitting_w128/checkpoint_* 2>/dev/null | wc -l)
        if [ $CKPT_COUNT -ge 50 ]; then
            echo "✓ COMPLETED ($CKPT_COUNT checkpoints)"
        else
            echo "⏸️  IDLE ($CKPT_COUNT checkpoints exist)"
        fi
    else
        echo "⏳ NOT STARTED"
    fi
    
    echo ""
    echo "=========================================="
    
    # Summary
    TOTAL_RUNNING=0
    TOTAL_COMPLETED=0
    TOTAL_CRASHED=0
    
    if [ "$W64_RUNNING" = true ]; then
        TOTAL_RUNNING=$((TOTAL_RUNNING + 1))
    elif [ -d "overfitting_w64" ]; then
        CKPT_COUNT=$(ls -1d overfitting_w64/checkpoint_* 2>/dev/null | wc -l)
        if [ $CKPT_COUNT -ge 50 ]; then
            TOTAL_COMPLETED=$((TOTAL_COMPLETED + 1))
        fi
    fi
    
    if [ "$W128_RUNNING" = true ]; then
        TOTAL_RUNNING=$((TOTAL_RUNNING + 1))
    elif [ -d "overfitting_w128" ]; then
        CKPT_COUNT=$(ls -1d overfitting_w128/checkpoint_* 2>/dev/null | wc -l)
        if [ $CKPT_COUNT -ge 50 ]; then
            TOTAL_COMPLETED=$((TOTAL_COMPLETED + 1))
        fi
    fi
    
    if [ $W64_CRASH_COUNT -gt 0 ]; then
        TOTAL_CRASHED=$((TOTAL_CRASHED + 1))
    fi
    if [ $W128_CRASH_COUNT -gt 0 ]; then
        TOTAL_CRASHED=$((TOTAL_CRASHED + 1))
    fi
    
    echo "Status Summary:"
    echo "  Running: $TOTAL_RUNNING"
    echo "  Completed: $TOTAL_COMPLETED"
    echo "  Crashed: $TOTAL_CRASHED"
    echo ""
    
    if [ $TOTAL_COMPLETED -eq 2 ]; then
        echo "🎉 ALL EXPERIMENTS COMPLETE!"
        echo ""
        echo "Run analysis:"
        echo "  python analyze_overfitting.py"
        echo ""
        break
    fi
    
    if [ $TOTAL_CRASHED -gt 0 ]; then
        echo "⚠️  ATTENTION: Crash detected!"
        echo "Check crash logs for details:"
        if [ $W64_CRASH_COUNT -gt 0 ]; then
            echo "  - overfitting_w64_crash.log"
        fi
        if [ $W128_CRASH_COUNT -gt 0 ]; then
            echo "  - overfitting_w128_crash.log"
        fi
        echo ""
    fi
    
    echo "Next check in 60 seconds..."
    echo "=========================================="
    
    # Update previous state
    PREV_W64_RUNNING=$W64_RUNNING
    PREV_W128_RUNNING=$W128_RUNNING
    
    sleep 60
done
