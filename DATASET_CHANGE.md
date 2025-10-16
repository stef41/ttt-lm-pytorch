# Dataset Change: WikiText-2 → C4

**Date**: October 16, 2025

## Summary

The default training dataset has been changed from **WikiText-2** to **C4 (Colossal Clean Crawled Corpus)**.

## Changes Made

### Updated Files

#### `tune.py`
- **Default dataset_name**: `"wikitext"` → `"allenai/c4"`
- **Default dataset_config**: `"wikitext-2-raw-v1"` → `"en"`

## Dataset Details

### C4 (Colossal Clean Crawled Corpus)
- **Source**: Web crawl data (Common Crawl)
- **Size**: ~750GB of cleaned English text
- **Configuration**: Using the English (`en`) subset
- **Provider**: AllenAI via Hugging Face datasets
- **Access**: `allenai/c4` on Hugging Face Hub

### Advantages of C4 over WikiText-2
1. **Scale**: Much larger dataset (~750GB vs ~4MB)
2. **Diversity**: Web-scale data from various domains
3. **Modern**: More recent and diverse content
4. **Common benchmark**: Widely used in recent LLM research

## Usage

### Default behavior (now uses C4):
```bash
python tune.py --trials 4 --search random --output-dir tuning_runs
```

### To use WikiText-2 (legacy):
```bash
python tune.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1
```

### To use WikiText-103:
```bash
python tune.py --dataset-name wikitext --dataset-config wikitext-103-raw-v1
```

### To use a different C4 language:
```bash
python tune.py --dataset-name allenai/c4 --dataset-config de  # German
python tune.py --dataset-name allenai/c4 --dataset-config es  # Spanish
```

## Notes

- The tokenizer remains unchanged (Llama-2-7b tokenizer)
- C4 requires more download time on first use due to its size
- Consider using `--dataset-subset-size` for faster experimentation
- The dataset will be cached by Hugging Face datasets library

## Requirements

Make sure you have the `datasets` library installed:
```bash
pip install datasets
```

For faster loading, you can also install:
```bash
pip install datasets[streaming]
```
