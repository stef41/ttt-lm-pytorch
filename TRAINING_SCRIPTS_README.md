# Training Scripts Summary

## 📁 Files Created

### Main Training Script
- **`train_ttt.py`** (11 KB)
  - Full-featured training script with all options
  - Supports all model sizes (125m, 350m, 760m, 1b)
  - Multi-GPU with Accelerate
  - W&B integration
  - Checkpointing and resumption

### Model-Specific Launch Scripts
- **`train_125m_c4.sh`** - Launch TTT-125M training
- **`train_350m_c4.sh`** - Launch TTT-350M training
- **`train_760m_c4.sh`** - Launch TTT-760M training
- **`train_1b_c4.sh`** - Launch TTT-1B training

### Interactive Launcher
- **`launch_training.sh`** - Menu-based training launcher

### Documentation
- **`QUICKSTART.md`** - Quick start guide
- **`TRAINING_RECIPES.md`** - Detailed training documentation
- **`DATASET_CHANGE.md`** - Dataset migration guide (WikiText-2 → C4)

## 🎯 Training Recipes (from JAX codebase)

All recipes follow the official configurations:

```
Common Settings:
├── Dataset: C4 English
├── Sequence Length: 2048 tokens
├── Global Batch Size: 256 sequences (0.5M tokens/step)
├── Weight Decay: 0.1
├── Optimizer: AdamW
├── Mixed Precision: BF16
└── TTT Layer: Linear (ttt_base_lr=1.0)

TTT-125M:
├── Steps: 4,800
├── Learning Rate: 3e-3 → 1e-5
├── Warmup: 480 steps (10%)
└── Total Tokens: ~2.4B

TTT-350M:
├── Steps: 13,500
├── Learning Rate: 1.5e-3 → 1e-5
├── Warmup: 1,350 steps (10%)
└── Total Tokens: ~6.75B

TTT-760M:
├── Steps: 29,000
├── Learning Rate: 1.25e-3 → 1e-5
├── Warmup: 2,900 steps (10%)
└── Total Tokens: ~14.5B

TTT-1B:
├── Steps: 50,000
├── Learning Rate: 1e-3 → 1e-5
├── Warmup: 5,000 steps (10%)
└── Total Tokens: ~25B
```

## 🚀 Usage Examples

### Quick Start
```bash
# Interactive launcher
./launch_training.sh

# Direct launch
./train_125m_c4.sh
```

### Custom Training
```bash
# Train with custom parameters
python train_ttt.py \
    --model_size "125m" \
    --dataset_name "allenai/c4" \
    --dataset_config "en" \
    --seq_length 2048 \
    --per_device_train_batch_size 32 \
    --max_train_steps 4800 \
    --learning_rate 3e-3 \
    --output_dir "./my_experiment"
```

### Resume Training
```bash
python train_ttt.py \
    --model_size "125m" \
    --resume_from_checkpoint "experiments/checkpoint-3000" \
    --max_train_steps 10000
```

## 🔧 Key Parameters

### Model Configuration
- `--model_size`: "125m", "350m", "760m", "1b"
- `--ttt_layer_type`: "linear", "mlp"
- `--ttt_base_lr`: TTT-specific learning rate (default: 1.0)

### Dataset Configuration
- `--dataset_name`: "allenai/c4" (default)
- `--dataset_config`: "en" (English), "de", "es", etc.
- `--seq_length`: Sequence length in tokens (default: 2048)

### Training Configuration
- `--per_device_train_batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Gradient accumulation
- `--max_train_steps`: Total training steps
- `--learning_rate`: Peak learning rate
- `--lr_end`: Final learning rate (default: 1e-5)
- `--lr_warmup_steps`: Warmup steps
- `--weight_decay`: AdamW weight decay (default: 0.1)
- `--mixed_precision`: "no", "fp16", "bf16" (default)

### Checkpointing
- `--save_checkpoint_freq`: Save every N steps (default: 1000)
- `--save_milestone_freq`: Save milestone every N steps (default: 2000)
- `--output_dir`: Output directory

### Logging
- `--logging_steps`: Log every N steps (default: 10)
- `--wandb_project`: W&B project name
- `--wandb_run_name`: W&B run name

## 📊 Expected Outputs

### During Training
```
***** Running training *****
  Max steps = 4800
  Per device batch size = 32
  Gradient accumulation steps = 1
  Total batch size = 256

Step 10: loss=8.2345, lr=6.25e-04
Step 20: loss=7.8912, lr=1.25e-03
...
Checkpoint saved to experiments/ttt_125m_c4_xxx/checkpoint-1000
```

### Directory Structure
```
experiments/ttt_125m_c4_20251016_123456/
├── checkpoint-1000/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── training_state.pt
├── milestone-2000/
├── checkpoint-3000/
├── final_model/
└── training_args.json
```

## 🎓 Best Practices

1. **Start Small**: Test with 125M before scaling up
2. **Monitor Closely**: Use W&B for real-time tracking
3. **Save Often**: Checkpoints every 1K steps (already configured)
4. **Test Recovery**: Verify checkpoint resumption works
5. **Check GPU Usage**: Ensure all GPUs are utilized (`nvidia-smi`)

## 🔍 Verification

Before starting a long training run:

```bash
# Test training for 10 steps
python train_ttt.py \
    --model_size "125m" \
    --max_train_steps 10 \
    --output_dir "./test_run"

# Verify checkpoint loading
python train_ttt.py \
    --model_size "125m" \
    --resume_from_checkpoint "./test_run/checkpoint-10" \
    --max_train_steps 20 \
    --output_dir "./test_run"
```

## 📈 Performance Tuning

### Memory Optimization
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use gradient checkpointing (requires code modification)

### Speed Optimization
- Use BF16 (enabled by default)
- Ensure fast storage for dataset cache
- Use multiple dataloading workers (requires code modification)

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce batch size, increase grad accumulation |
| Slow training | Check GPU utilization, verify BF16 enabled |
| Dataset download fails | Check internet, set HF_DATASETS_CACHE |
| Checkpoint not found | Verify path, check permissions |
| W&B not logging | Run `wandb login`, check project name |

## 📚 Related Documentation

- **README.md** - General project information
- **QUICKSTART.md** - Quick start guide
- **TRAINING_RECIPES.md** - Detailed training guide
- **DATASET_CHANGE.md** - Dataset information

## 🎯 Next Steps

1. ✅ Review `QUICKSTART.md` for quick start
2. ✅ Read `TRAINING_RECIPES.md` for details
3. ✅ Run `./launch_training.sh` to start training
4. ✅ Monitor training on W&B dashboard
5. ✅ Evaluate trained models on your tasks

---

**Happy Training!** 🚀
