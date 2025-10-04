#!/bin/bash
# Wait for curriculum training to complete, then generate plots and summary

CURRICULUM_DIR="curriculum_exp"

echo "⏳ Waiting for curriculum training to complete..."

# Wait loop
while true; do
    checkpoint_count=$(find "${CURRICULUM_DIR}" -maxdepth 1 -type d -name "checkpoint_*" 2>/dev/null | wc -l)
    
    if [ "$checkpoint_count" -ge 30 ]; then
        echo "✅ All 30 checkpoints detected!"
        break
    fi
    
    echo "$(date): $checkpoint_count/30 checkpoints complete..."
    sleep 300  # Check every 5 minutes
done

# Wait a bit more to ensure final model is saved
sleep 60

echo ""
echo "🎨 Generating evolution plots..."
cd /data/users/zacharie/ttt-lm-pytorch

python plot_curriculum_results.py --curriculum_dir curriculum_exp

echo ""
echo "📊 Final Summary:"
python check_curriculum_status.py

echo ""
echo "🎉 All done! Results saved in curriculum_exp/"
echo "   - Main plot: curriculum_exp/curriculum_length_gen_evolution.png"
echo "   - Individual checkpoints: curriculum_exp/checkpoint_XXX/"
echo "   - Training log: curriculum_exp/training.log"
