#!/bin/bash
# Monitor overfitting study progress

clear
echo "=========================================="
echo "TTT OVERFITTING STUDY MONITOR"
echo "=========================================="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check which experiments are running/completed
W64_RUNNING=false
W128_RUNNING=false

if pgrep -f "train_overfitting_study.py.*overfitting_w64" > /dev/null; then
    W64_RUNNING=true
fi

if pgrep -f "train_overfitting_study.py.*overfitting_w128" > /dev/null; then
    W128_RUNNING=true
fi

echo "=========================================="
echo "EXPERIMENT STATUS"
echo "=========================================="

# W=64 Status
echo ""
echo "W=64 Experiment:"
echo "----------------"
if [ "$W64_RUNNING" = true ]; then
    PID=$(pgrep -f "train_overfitting_study.py.*overfitting_w64")
    echo "Status: ✓ RUNNING (PID: $PID)"
    
    if [ -f "overfitting_w64_training.log" ]; then
        echo ""
        echo "Recent progress:"
        tail -20 overfitting_w64_training.log | grep -E "(Training:|Step|Checkpoint|@1xW|@10xW)" | tail -10
    fi
    
    if [ -d "overfitting_w64" ]; then
        CKPT_COUNT=$(ls -1d overfitting_w64/checkpoint_* 2>/dev/null | wc -l)
        echo ""
        echo "Checkpoints completed: $CKPT_COUNT / 50"
        
        if [ $CKPT_COUNT -gt 0 ]; then
            LATEST=$(ls -1d overfitting_w64/checkpoint_* | tail -1)
            echo "Latest checkpoint: $LATEST"
            if [ -f "$LATEST/summary.json" ]; then
                python3 << 'EOF'
import json
with open("$LATEST/summary.json") as f:
    data = json.load(f)
print(f"  Step: {data['step']:,}")
print(f"  TTT@1×W: {data['ttt_ppl_at_1xW']:.1f}")
print(f"  TTT@10×W: {data.get('ttt_ppl_at_10xW', 'N/A')}")
EOF
            fi
        fi
    fi
    
    # Estimate completion
    if [ -f "overfitting_w64_training.log" ]; then
        CURRENT_STEP=$(tail -100 overfitting_w64_training.log | grep "Training:" | tail -1 | sed 's/.*Training: *\([0-9]*\)%.*/\1/')
        if [ ! -z "$CURRENT_STEP" ]; then
            echo ""
            echo "Progress: ${CURRENT_STEP}% of 100k steps"
            REMAINING=$((100 - CURRENT_STEP))
            echo "Estimated remaining: ~$(($REMAINING * 4 / 60)) hours"
        fi
    fi
    
elif [ -d "overfitting_w64" ]; then
    echo "Status: ✓ COMPLETED"
    CKPT_COUNT=$(ls -1d overfitting_w64/checkpoint_* 2>/dev/null | wc -l)
    echo "Total checkpoints: $CKPT_COUNT"
    
    if [ -f "overfitting_w64/checkpoint_$(printf '%03d' $((CKPT_COUNT-1)))/summary.json" ]; then
        LATEST="overfitting_w64/checkpoint_$(printf '%03d' $((CKPT_COUNT-1)))"
        echo ""
        echo "Final checkpoint: $LATEST"
        python3 -c "
import json
with open('$LATEST/summary.json') as f:
    data = json.load(f)
print(f\"  Step: {data['step']:,}\")
print(f\"  TTT@1×W: {data['ttt_ppl_at_1xW']:.1f}\")
if 'ttt_ppl_at_10xW' in data:
    print(f\"  TTT@10×W: {data['ttt_ppl_at_10xW']:.1f}\")
    print(f\"  SW@10×W: {data['sw_ppl_at_10xW']:.1f}\")
    print(f\"  SW/TTT ratio: {data['sw_ppl_at_10xW']/data['ttt_ppl_at_10xW']:.3f}\")
"
    fi
else
    echo "Status: ✗ NOT STARTED"
fi

# W=128 Status
echo ""
echo "W=128 Experiment:"
echo "-----------------"
if [ "$W128_RUNNING" = true ]; then
    PID=$(pgrep -f "train_overfitting_study.py.*overfitting_w128")
    echo "Status: ✓ RUNNING (PID: $PID)"
    
    if [ -f "overfitting_w128_training.log" ]; then
        echo ""
        echo "Recent progress:"
        tail -20 overfitting_w128_training.log | grep -E "(Training:|Step|Checkpoint|@1xW|@10xW)" | tail -10
    fi
    
    if [ -d "overfitting_w128" ]; then
        CKPT_COUNT=$(ls -1d overfitting_w128/checkpoint_* 2>/dev/null | wc -l)
        echo ""
        echo "Checkpoints completed: $CKPT_COUNT / 50"
        
        if [ $CKPT_COUNT -gt 0 ]; then
            LATEST=$(ls -1d overfitting_w128/checkpoint_* | tail -1)
            echo "Latest checkpoint: $LATEST"
        fi
    fi
    
    # Estimate completion
    if [ -f "overfitting_w128_training.log" ]; then
        CURRENT_STEP=$(tail -100 overfitting_w128_training.log | grep "Training:" | tail -1 | sed 's/.*Training: *\([0-9]*\)%.*/\1/')
        if [ ! -z "$CURRENT_STEP" ]; then
            echo ""
            echo "Progress: ${CURRENT_STEP}% of 100k steps"
            REMAINING=$((100 - CURRENT_STEP))
            echo "Estimated remaining: ~$(($REMAINING * 4 / 60)) hours"
        fi
    fi
    
elif [ -d "overfitting_w128" ]; then
    echo "Status: ✓ COMPLETED"
    CKPT_COUNT=$(ls -1d overfitting_w128/checkpoint_* 2>/dev/null | wc -l)
    echo "Total checkpoints: $CKPT_COUNT"
    
    if [ -f "overfitting_w128/checkpoint_$(printf '%03d' $((CKPT_COUNT-1)))/summary.json" ]; then
        LATEST="overfitting_w128/checkpoint_$(printf '%03d' $((CKPT_COUNT-1)))"
        echo ""
        echo "Final checkpoint: $LATEST"
        python3 -c "
import json
with open('$LATEST/summary.json') as f:
    data = json.load(f)
print(f\"  Step: {data['step']:,}\")
print(f\"  TTT@1×W: {data['ttt_ppl_at_1xW']:.1f}\")
if 'ttt_ppl_at_10xW' in data:
    print(f\"  TTT@10×W: {data['ttt_ppl_at_10xW']:.1f}\")
    print(f\"  SW@10×W: {data['sw_ppl_at_10xW']:.1f}\")
    print(f\"  SW/TTT ratio: {data['sw_ppl_at_10xW']/data['ttt_ppl_at_10xW']:.3f}\")
"
    fi
else
    echo "Status: ✗ NOT STARTED"
fi

echo ""
echo "=========================================="
echo "OVERALL PROGRESS"
echo "=========================================="

COMPLETED=0
RUNNING=0
NOT_STARTED=0

if [ -d "overfitting_w64" ] && [ "$W64_RUNNING" = false ]; then
    COMPLETED=$((COMPLETED + 1))
elif [ "$W64_RUNNING" = true ]; then
    RUNNING=$((RUNNING + 1))
else
    NOT_STARTED=$((NOT_STARTED + 1))
fi

if [ -d "overfitting_w128" ] && [ "$W128_RUNNING" = false ]; then
    COMPLETED=$((COMPLETED + 1))
elif [ "$W128_RUNNING" = true ]; then
    RUNNING=$((RUNNING + 1))
else
    NOT_STARTED=$((NOT_STARTED + 1))
fi

echo "Experiments completed: $COMPLETED / 2"
echo "Experiments running: $RUNNING"
echo "Experiments pending: $NOT_STARTED"

if [ $COMPLETED -eq 2 ]; then
    echo ""
    echo "🎉 ALL EXPERIMENTS COMPLETE!"
    echo ""
    echo "To analyze results, run:"
    echo "  python analyze_overfitting.py"
fi

echo ""
echo "=========================================="
echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
