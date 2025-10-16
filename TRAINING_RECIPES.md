# TTT Training Recipes on C4

Training scripts based on the official JAX training configurations, adapted for PyTorch.

## Model Configurations

All models are trained with:
- **Dataset**: C4 (Colossal Clean Crawled Corpus)
- **Sequence Length**: 2048 tokens
- **Global Batch Size**: 256 sequences (0.5M tokens per step)
- **Optimizer**: AdamW with weight decay 0.1
- **TTT Layer Type**: Linear (ttt_linear)
- **TTT Base LR**: 1.0
- **Mixed Precision**: BF16

### TTT-125M
- **Parameters**: ~125 million
- **Training Steps**: 4,800
- **Learning Rate**: 3e-3 → 1e-5 (cosine decay)
- **Warmup Steps**: 480 (10% of total)
- **Total Tokens**: ~2.4 billion

### TTT-350M
- **Parameters**: ~350 million
- **Training Steps**: 13,500
- **Learning Rate**: 1.5e-3 → 1e-5 (cosine decay)
- **Warmup Steps**: 1,350 (10% of total)
- **Total Tokens**: ~6.75 billion

### TTT-760M
- **Parameters**: ~760 million
- **Training Steps**: 29,000
- **Learning Rate**: 1.25e-3 → 1e-5 (cosine decay)
- **Warmup Steps**: 2,900 (10% of total)
- **Total Tokens**: ~14.5 billion

### TTT-1B
- **Parameters**: ~1 billion
- **Training Steps**: 50,000
- **Learning Rate**: 1e-3 → 1e-5 (cosine decay)
- **Warmup Steps**: 5,000 (10% of total)
- **Total Tokens**: ~25 billion

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install transformers accelerate datasets torch wandb

# Or use requirements.txt
pip install -r requirements.txt
```

### Single Model Training

```bash
# Train TTT-125M
chmod +x train_125m_c4.sh
./train_125m_c4.sh

# Train TTT-350M
chmod +x train_350m_c4.sh
./train_350m_c4.sh

# Train TTT-760M
chmod +x train_760m_c4.sh
./train_760m_c4.sh

# Train TTT-1B
chmod +x train_1b_c4.sh
./train_1b_c4.sh
```

### Custom Training

You can also use the `train_ttt.py` script directly:

```bash
accelerate launch --config_file accelerate_config.yaml train_ttt.py \
    --model_size "125m" \
    --dataset_name "allenai/c4" \
    --dataset_config "en" \
    --seq_length 2048 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 4800 \
    --learning_rate 3e-3 \
    --lr_end 1e-5 \
    --lr_warmup_steps 480 \
    --weight_decay 0.1 \
    --mixed_precision "bf16" \
    --state_passing \
    --output_dir "./experiments/my_experiment"
```

## Multi-GPU Training

The scripts automatically detect available GPUs and calculate appropriate batch sizes:

### 8 GPUs (Recommended)
- Per-device batch size: 32
- Gradient accumulation: 1
- Effective batch size: 256

### 4 GPUs
- Per-device batch size: 32
- Gradient accumulation: 2
- Effective batch size: 256

### 2 GPUs
- Per-device batch size: 16
- Gradient accumulation: 8
- Effective batch size: 256

### Single GPU
- Per-device batch size: 16
- Gradient accumulation: 16
- Effective batch size: 256

## Accelerate Configuration

Make sure your `accelerate_config.yaml` is properly configured:

```bash
accelerate config
```

Or use the provided configuration for multi-GPU training:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_machines: 1
num_processes: 8  # Set to your number of GPUs
use_cpu: false
```

## Monitoring with Weights & Biases

All training scripts support W&B logging. To enable:

1. Install W&B: `pip install wandb`
2. Login: `wandb login`
3. Training automatically logs to the `ttt-lm-pytorch` project

The scripts log:
- Training loss
- Learning rate
- Step count
- Model checkpoints

## Checkpointing

### Regular Checkpoints
Saved every 1,000 steps at: `experiments/{exp_name}/checkpoint-{step}/`

### Milestone Checkpoints
Saved every 2,000 steps at: `experiments/{exp_name}/milestone-{step}/`

### Final Model
Saved at completion: `experiments/{exp_name}/final_model/`

Each checkpoint includes:
- Model weights
- Tokenizer
- Optimizer state
- Scheduler state
- Training arguments

## Resume Training

To resume from a checkpoint:

```bash
python train_ttt.py \
    --resume_from_checkpoint "experiments/ttt_125m_c4_20251016/checkpoint-3000" \
    [other args...]
```

## Hardware Requirements

### Estimated GPU Memory (BF16)

| Model | Single GPU (no grad accum) | With Gradient Accumulation |
|-------|---------------------------|---------------------------|
| 125M  | ~12 GB                    | ~8 GB                     |
| 350M  | ~20 GB                    | ~12 GB                    |
| 760M  | ~40 GB                    | ~24 GB                    |
| 1B    | ~50 GB                    | ~32 GB                    |

### Recommended Setup

- **125M**: 1-2 GPUs (A100/V100)
- **350M**: 2-4 GPUs (A100)
- **760M**: 4-8 GPUs (A100 40GB/80GB)
- **1B**: 8 GPUs (A100 80GB)

## Training Time Estimates

On 8x A100 80GB GPUs:

| Model | Steps | Time per Step | Total Time |
|-------|-------|--------------|------------|
| 125M  | 4,800 | ~2s          | ~2.7 hours |
| 350M  | 13,500| ~3s          | ~11 hours  |
| 760M  | 29,000| ~5s          | ~40 hours  |
| 1B    | 50,000| ~7s          | ~97 hours  |

*Times are approximate and vary based on hardware and configuration.*

## Dataset Information

### C4 (Colossal Clean Crawled Corpus)

- **Source**: Common Crawl web scrape
- **Size**: ~750 GB of cleaned English text
- **Tokens**: ~156 billion tokens (with Llama-2 tokenizer)
- **Processing**: Filtered for quality, deduplicated
- **Streaming**: Uses HuggingFace datasets streaming to avoid downloading entire dataset

### First Run
The first time you run training, C4 will be downloaded and cached. This may take some time depending on your internet connection. Subsequent runs will use the cached data.

## Troubleshooting

### Out of Memory (OOM)
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Use smaller model size
4. Enable gradient checkpointing (requires modifying model)

### Slow Training
1. Ensure you're using BF16 mixed precision
2. Check if dataset is cached (first run is slower)
3. Verify all GPUs are being utilized
4. Use faster storage for dataset cache

### Dataset Download Issues
```bash
# Set cache directory
export HF_DATASETS_CACHE="/path/to/large/storage"

# Use offline mode after first download
export HF_DATASETS_OFFLINE=1
```

## Differences from JAX Implementation

This PyTorch implementation differs from the original JAX code in:

1. **No mesh dimensions**: Uses Accelerate instead of JAX's mesh parallelism
2. **Streaming dataset**: C4 is streamed to save disk space
3. **Checkpoint format**: Uses HuggingFace format instead of JAX checkpoints
4. **Logging**: Integrated with W&B and standard Python logging

## References

- Original Paper: [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620)
- JAX Codebase: https://github.com/test-time-training/ttt-lm-jax
- C4 Dataset: https://huggingface.co/datasets/allenai/c4

## License

Same as the main repository (see LICENSE file).
