# TTT Overfitting Study - Experiment Plan

## Objective
Study overfitting behavior by training TTT models for full 100k steps **WITHOUT early stopping** for both W=64 and W=128 window sizes.

## Experiments

### Experiment 1: W=64 (Training Window = 64 tokens)
- **Status**: 🟢 RUNNING (Started: 2025-10-04 01:57)
- **Output**: `overfitting_w64/`
- **Log**: `overfitting_w64_training.log`
- **Max steps**: 100,000 (no early stopping)
- **Eval checkpoints**: 50 (every 2,040 steps)
- **Estimated time**: ~6-8 hours
- **Current phase**: Dataset tokenization

### Experiment 2: W=128 (Training Window = 128 tokens)
- **Status**: ⏳ PENDING (will start after W=64 completes)
- **Output**: `overfitting_w128/`
- **Log**: `overfitting_w128_training.log`
- **Max steps**: 100,000 (no early stopping)
- **Eval checkpoints**: 50 (every 2,040 steps)
- **Estimated time**: ~6-8 hours

## Key Differences from Previous Experiments

### Previous Experiments (with early stopping):
- W=64: Stopped at 15,000 steps (undertrained)
- W=128: Stopped at ~45,000 steps (early stopping triggered)

### Current Experiments (NO early stopping):
- Both W=64 and W=128 will run for **full 100,000 steps**
- This allows studying:
  1. **Convergence patterns**: When does the model reach its best performance?
  2. **Overfitting detection**: Does PPL start increasing after optimal point?
  3. **Window size effects**: Does larger window (W=128) overfit slower/faster?
  4. **Length generalization degradation**: Does overfitting affect long-context more?

## Evaluation Metrics

At each checkpoint, we evaluate:
1. **TTT PPL @ 1×W** (training window): Measures in-distribution performance
2. **SW PPL @ 1×W**: Sliding window baseline at training window
3. **TTT PPL @ 10×W** (10× training window): Length generalization capability
4. **SW PPL @ 10×W**: Baseline for long context
5. **TTT PPL @ 20×W** (20× training window): Extreme length generalization
6. **SW PPL @ 20×W**: Baseline for extreme length

## Expected Findings

### Overfitting Indicators:
- **Training PPL (1×W)** continues decreasing
- **Validation/long-context PPL (10×W, 20×W)** starts increasing
- Gap between training and long-context PPL widens

### Window Size Effects:
- **Hypothesis 1**: Larger window (W=128) may overfit slower due to more diverse training data
- **Hypothesis 2**: W=128 may maintain better length generalization even with overfitting
- **Hypothesis 3**: Optimal stopping point differs between W=64 and W=128

## Monitoring

To check experiment status:
```bash
./monitor_overfitting.sh
```

This shows:
- Current training step and progress %
- Number of evaluation checkpoints completed
- Latest perplexity metrics
- Estimated time remaining

## Analysis

After both experiments complete, run:
```bash
python analyze_overfitting.py
```

This will generate:
- `overfitting_analysis.png`: Comprehensive visualization showing:
  - Training curves for both window sizes
  - Overfitting detection (marking when PPL starts degrading)
  - SW vs TTT comparison at different sequence lengths
  - Summary tables with best checkpoints and overfitting indicators

## Timeline

- **Start**: October 4, 2025, 01:57 UTC
- **W=64 Expected Completion**: October 4, 2025, ~08:00-10:00 UTC
- **W=128 Expected Completion**: October 4, 2025, ~14:00-18:00 UTC
- **Total Duration**: ~12-16 hours

## Comparison with Previous Results

### Previous W=64 (15k steps, early stopped):
- Final TTT PPL@1×W: 3,153.5
- Final TTT PPL@10×W: 3,222.0
- SW/TTT@10×W ratio: 0.855 (SW wins by 14.5%)

### Previous W=128 (45k steps, early stopped):
- Best TTT PPL@1×W: 240.3 at step 24,489
- Final TTT PPL@1×W: 215.8 at step 44,897
- Final TTT PPL@10×W: 338.9
- SW/TTT@10×W ratio: 0.968 (TTT wins by 3.2%)

### Questions to Answer:
1. Would W=64 reach similar PPL as W=128 with more training?
2. At what step does each model reach optimal performance?
3. Does training beyond optimal point harm length generalization?
4. Can we establish a clear relationship between window size and overfitting?

## Files Created

- `train_overfitting_study.py`: Training script without early stopping
- `run_overfitting_study.sh`: Launcher for both experiments
- `monitor_overfitting.sh`: Real-time monitoring script
- `analyze_overfitting.py`: Post-training analysis and visualization
- `OVERFITTING_STUDY.md`: This document
