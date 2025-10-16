# Quick Start Guide - TTT Training on C4

## 🚀 Quick Launch

```bash
# Interactive launcher
./launch_training.sh

# Or directly launch specific model
./train_125m_c4.sh   # TTT-125M
./train_350m_c4.sh   # TTT-350M
./train_760m_c4.sh   # TTT-760M
./train_1b_c4.sh     # TTT-1B
```

## 📋 Training Configurations

| Model   | Steps  | LR     | Warmup | Tokens  | Time (8xA100) |
|---------|--------|--------|--------|---------|---------------|
| 125M    | 4.8K   | 3e-3   | 480    | 2.4B    | ~2.7 hours    |
| 350M    | 13.5K  | 1.5e-3 | 1350   | 6.75B   | ~11 hours     |
| 760M    | 29K    | 1.25e-3| 2900   | 14.5B   | ~40 hours     |
| 1B      | 50K    | 1e-3   | 5000   | 25B     | ~97 hours     |

## 📦 Prerequisites

```bash
# Install dependencies
pip install transformers accelerate datasets torch wandb

# Configure accelerate for your setup
accelerate config

# Login to W&B (optional but recommended)
wandb login
```

## 🎯 Key Features

- ✅ **Official Training Recipes**: Based on JAX implementation
- ✅ **C4 Dataset**: Web-scale training data
- ✅ **Multi-GPU Support**: Automatic GPU detection and scaling
- ✅ **W&B Integration**: Real-time monitoring
- ✅ **Checkpointing**: Save every 1K steps, milestones every 2K
- ✅ **BF16 Mixed Precision**: Faster training, lower memory
- ✅ **Streaming Dataset**: No need to download 750GB upfront

## 🔧 Customization

Edit the shell scripts to customize:
- `DATASET_NAME`: Change dataset
- `DATASET_CONFIG`: Change language or subset
- `SEQ_LEN`: Change sequence length
- `BATCH_SIZE`: Adjust batch size
- `EXP_DIR`: Change output directory

Or use `train_ttt.py` directly for full control.

## 📊 Monitoring

Training metrics are logged to:
- **Console**: Every 10 steps
- **W&B Dashboard**: Real-time (if enabled)
- **Local Files**: In `experiments/{exp_name}/`

## 💾 Output Structure

```
experiments/
└── ttt_125m_c4_20251016_123456/
    ├── checkpoint-1000/
    ├── checkpoint-2000/
    ├── milestone-2000/
    ├── checkpoint-4000/
    ├── milestone-4000/
    ├── final_model/
    └── training_args.json
```

## 🐛 Troubleshooting

**Out of Memory?**
- Reduce `PER_DEVICE_BS` in the script
- Increase `GRAD_ACCUM_STEPS` accordingly

**Slow Download?**
- First run downloads C4 (takes time)
- Subsequent runs use cached data
- Set `HF_DATASETS_CACHE` for custom cache location

**Multi-GPU Issues?**
- Run `accelerate config` and follow prompts
- Verify GPUs visible: `nvidia-smi`

## 📚 Documentation

- **Full Guide**: See `TRAINING_RECIPES.md`
- **Dataset Info**: See `DATASET_CHANGE.md`
- **Original Paper**: https://arxiv.org/abs/2407.04620

## 🎓 Example Usage

```bash
# Train 125M model on 4 GPUs
NUM_GPUS=4 ./train_125m_c4.sh

# Resume from checkpoint
python train_ttt.py \
    --model_size "125m" \
    --resume_from_checkpoint "experiments/ttt_125m_c4_xxx/checkpoint-3000" \
    [other args...]

# Train with custom settings
python train_ttt.py \
    --model_size "350m" \
    --seq_length 4096 \
    --learning_rate 2e-3 \
    --max_train_steps 20000 \
    --output_dir "./my_experiment"
```

## ⚡ Tips

1. **Use BF16**: Much faster than FP32, enabled by default
2. **Monitor W&B**: Set `--wandb_project` flag
3. **Save Checkpoints**: Auto-saved every 1K steps
4. **Test First**: Run a few steps to verify setup before long training
5. **Cache Dataset**: First run is slower due to download

## 🤝 Support

For issues or questions:
- Check `TRAINING_RECIPES.md` for detailed documentation
- Review the original JAX codebase for reference
- Open an issue on GitHub

---

**Ready to train?** Run `./launch_training.sh` to get started! 🎉
