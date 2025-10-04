#!/bin/bash
# Quick status check and useful commands for overfitting study

echo "TTT OVERFITTING STUDY - QUICK REFERENCE"
echo "========================================"
echo ""

echo "📊 CHECK STATUS:"
echo "  ./monitor_overfitting.sh"
echo ""

echo "📝 VIEW LOGS:"
echo "  tail -f overfitting_w64_training.log    # W=64 live log"
echo "  tail -f overfitting_w128_training.log   # W=128 live log"
echo ""

echo "🔍 CHECK PROCESS:"
echo "  ps aux | grep train_overfitting_study"
echo ""

echo "📈 VIEW RECENT PROGRESS:"
echo "  tail -100 overfitting_w64_training.log | grep 'Training:'"
echo "  tail -100 overfitting_w128_training.log | grep 'Training:'"
echo ""

echo "📊 COUNT CHECKPOINTS:"
echo "  ls -1d overfitting_w64/checkpoint_* | wc -l"
echo "  ls -1d overfitting_w128/checkpoint_* | wc -l"
echo ""

echo "🎯 VIEW LATEST RESULTS:"
echo "  python3 << 'EOF'"
echo "import json, glob"
echo "w64_ckpts = sorted(glob.glob('overfitting_w64/checkpoint_*/summary.json'))"
echo "if w64_ckpts:"
echo "    with open(w64_ckpts[-1]) as f:"
echo "        d = json.load(f)"
echo "    print(f\"W=64 Latest: Step {d['step']:,}, TTT@1xW={d['ttt_ppl_at_1xW']:.1f}\")"
echo "EOF"
echo ""

echo "📊 ANALYZE RESULTS (after completion):"
echo "  python analyze_overfitting.py"
echo ""

echo "🛑 STOP TRAINING (if needed):"
echo "  pkill -f train_overfitting_study"
echo ""

echo "========================================"
echo "Current Status:"
./monitor_overfitting.sh | grep -A 5 "OVERALL PROGRESS"
