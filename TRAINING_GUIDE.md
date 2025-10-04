# Training TTT Models with HuggingFace Datasets

This guide shows how to train TTT (Test-Time Training) models on datasets from the HuggingFace datasets library with GPU acceleration.

## Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install tensorboard  # or wandb for monitoring
```

### Basic Training Examples

#### 1. Train on WikiText-2 (Small Dataset)

```bash
python train_gpu_optimized.py \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --model_size 125m \
    --ttt_layer_type linear \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-4 \
    --max_train_steps 1000 \
    --max_seq_length 1024 \
    --output_dir ./outputs/ttt-125m-wikitext2 \
    --mixed_precision bf16 \
    --logging_steps 100
```

#### 2. Train Larger Model

```bash
python train_gpu_optimized.py \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --model_size 350m \
    --ttt_layer_type mlp \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-4 \
    --max_train_steps 5000 \
    --max_seq_length 1024 \
    --output_dir ./outputs/ttt-760m-openwebtext \
    --mixed_precision bf16 \
    --output_dir ./outputs/ttt-350m-wikitext2 \
    --logging_steps 50
```

#### 3. Multi-GPU Training

```bash
# Use the provided accelerate config for multi-GPU training
accelerate launch --config_file accelerate_config.yaml train_gpu_optimized.py \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --model_size 350m \
    --ttt_layer_type linear \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --max_train_steps 2000 \
    --max_seq_length 1024 \
    --output_dir ./outputs/ttt-350m-multi-gpu \
    --mixed_precision bf16 \
    --logging_steps 100
```

## Model Configurations

### Available Model Sizes

| Size | Hidden Size | Layers | Attention Heads | Parameters |
|------|------------|--------|-----------------|------------|
| 125m | 768        | 12     | 12              | ~125M      |
| 350m | 1024       | 24     | 16              | ~350M      |
| 760m | 1536       | 24     | 16              | ~760M      |
| 1b   | 2048       | 24     | 32              | ~1B        |

### TTT-Specific Parameters

- `--ttt_layer_type`: Choose between `linear` (TTT-Linear) or `mlp` (TTT-MLP)
- `--ttt_base_lr`: Base learning rate for TTT learner (default: 1.0)
- `--use_gate`: Enable gating in Mamba backbone
- `--share_qk`: Share Q/K projection matrices
- `--pre_conv`: Use convolution before TTT layers
- `--conv_kernel`: Kernel size for convolution (default: 4)

## Dataset Support

The training script automatically handles various text datasets from HuggingFace:

### Text Column Detection

The script automatically detects text columns named:
- `text`
- `content`
- `article`
- `body`

### Supported Datasets

| Dataset | Command | Description |
|---------|---------|-------------|
| WikiText-2 | `--dataset_name wikitext --dataset_config wikitext-2-raw-v1` | Small Wikipedia dataset |
| WikiText-103 | `--dataset_name wikitext --dataset_config wikitext-103-raw-v1` | Large Wikipedia dataset |
| OpenWebText | `--dataset_name openwebtext` | Open reproduction of WebText |
| C4 | `--dataset_name c4 --dataset_config en` | Common Crawl clean text |
| BookCorpus | `--dataset_name bookcorpus` | Books dataset |
| The Pile | `--dataset_name the_pile` | Large diverse text dataset |

### Custom Datasets

For custom datasets, ensure they have a text column:

```python
# Example custom dataset structure
{
    "text": "Your training text here...",
    # other columns are ignored
}
```

## GPU Training Optimization

### Memory Optimization

For large models or limited GPU memory:

```bash
# Reduce batch size and increase gradient accumulation
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 64 \

# Use bfloat16 for better numerical stability
--mixed_precision bf16 \

# Reduce sequence length
--max_seq_length 1024
```

### Multi-GPU Setup

1. Use the provided accelerate config:
```bash
# Multi-GPU config is already provided in accelerate_config.yaml
```

2. Launch training:
```bash
accelerate launch --config_file accelerate_config.yaml train_gpu_optimized.py [args...]
```

### Recommended GPU Memory Requirements

| Model Size | Min GPU Memory | Recommended GPU Memory | Batch Size |
|------------|---------------|----------------------|------------|
| 125m       | 8GB           | 16GB                 | 4-8        |
| 350m       | 16GB          | 24GB                 | 2-4        |
| 760m       | 24GB          | 40GB                 | 1-2        |
| 1b         | 32GB          | 48GB                 | 1          |

## Monitoring and Logging

### TensorBoard (Default)

```bash
# Start training with built-in logging
python train_gpu_optimized.py [other args...]

# Logs are displayed in console and saved to output directory
```

### Console Logging

The train_gpu_optimized.py script includes comprehensive console logging:
- Real-time loss and learning rate
- GPU memory usage monitoring  
- Tokens/second throughput
- Training progress and timing

Example training command:
```bash
python train_gpu_optimized.py --max_train_steps 1000 --logging_steps 50 [other args...]
```

## Advanced Training Strategies

### Curriculum Learning

Start with shorter sequences and gradually increase:

```bash
# Start with basic training - the script includes optimized settings
python train_gpu_optimized.py \
    --model_size 125m \
    --max_seq_length 512 \
    --max_train_steps 1000 \
    --per_device_train_batch_size 8 \
    --mixed_precision bf16

# Phase 2: Medium sequences  
python train_ttt.py --max_seq_length 1024 --max_train_steps 20000 \
    --resume_from_checkpoint ./outputs/checkpoint-10000 [args...]

# Phase 3: Full sequences
python train_ttt.py --max_seq_length 2048 --max_train_steps 50000 \
    --resume_from_checkpoint ./outputs/checkpoint-20000 [args...]
```

### Learning Rate Schedules

```bash
# Linear warmup with decay
--learning_rate 5e-4 --warmup_ratio 0.05 --lr_scheduler_type linear

# For longer training, use lower learning rates
--learning_rate 2e-4 --warmup_ratio 0.03
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or sequence length
2. **Slow Training**: The train_gpu_optimized.py script is optimized for speed
3. **CUDA Errors**: Ensure CUDA and PyTorch versions are compatible
4. **Loss Spikes**: The script includes gradient clipping and stable training settings

### Performance Tips

1. **Use bf16 mixed precision**: Enabled by default in train_gpu_optimized.py
2. **Multi-GPU training**: Use accelerate_config.yaml for multi-GPU setup
3. **Batch size optimization**: Script automatically monitors GPU memory usage
4. **Monitoring**: Built-in GPU memory and speed monitoring

## Quick Test

To verify everything works:

```bash
# Quick test run (125M model, 50 steps)
python train_gpu_optimized.py \
    --model_size 125m \
    --max_train_steps 50 \
    --per_device_train_batch_size 4 \
    --logging_steps 10
```

## Repository Structure

```
ttt-lm-pytorch/
├── ttt.py                    # Core TTT implementation
├── train_gpu_optimized.py    # Optimized training script
├── requirements.txt          # Dependencies
├── accelerate_config.yaml    # Multi-GPU configuration
├── README.md                 # Project documentation
├── TRAINING_GUIDE.md         # This guide
└── LICENSE                   # License
```

## Citation

If you use this code, please cite the original TTT paper:

```bibtex
@article{sun2024learning,
  title={Learning to (Learn at Test Time): RNNs with Expressive Hidden States},
  author={Sun, Yu and Cazenavette, George and Dai, Zhiding and Kautz, Jan and Pavlovic, Vladimir and Yuan, Sifei},
  journal={arXiv preprint arXiv:2407.04620},
  year={2024}
}
```