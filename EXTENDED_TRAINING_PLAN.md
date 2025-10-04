# Extended Curriculum Training - 100k Steps

**Status:** 🟢 **RUNNING**  
**Started:** October 2, 2025  
**Expected Duration:** ~5-7 hours (est.)

---

## 🎯 Objective

Train the TTT-125M model to **full convergence** with a much longer training run to properly study length generalization emergence.

**Why Extend?**
- Previous 15k-step run showed model was **still improving** at the end
- Final PPL ~3k-4k is still high, indicating undertraining
- Loss was dropping steadily throughout (5.8 → still decreasing)
- Need to reach true plateau to see final length generalization behavior

---

## 📊 Configuration Changes

### Training Duration
- **Previous:** 15,000 steps (~42 min)
- **New:** 100,000 steps (~5-7 hours)
- **Improvement:** 6.7× longer training

### Evaluation Granularity
- **Previous:** 30 checkpoints
- **New:** 50 checkpoints
- **Improvement:** More fine-grained analysis of emergence

### Early Stopping
- **New Feature:** Stop if no improvement for 10 checkpoints
- **Min Delta:** 50 PPL improvement required
- **Purpose:** Avoid unnecessary computation if converged early

---

## 🔬 Expected Improvements

### Training Convergence
- **Target:** Reach PPL < 100 @ 1×W (vs current ~3k)
- **Metric:** Training loss should plateau completely
- **Validation:** Early stopping will trigger when converged

### Length Generalization
With proper convergence, we expect to see:

1. **Clearer Patterns:** More stable PPL values at long positions
2. **Better Comparison:** TTT vs SW with both fully trained
3. **True Crossover:** Determine if SW advantage at 10×W persists
4. **Extreme Range:** Better evaluation at 20×W position

---

## 📈 Training Parameters

```python
Model: TTTForCausalLM-125M
├── Hidden: 768
├── Layers: 12
├── Heads: 12
└── Params: 152.6M

Training:
├── Window (W): 64 tokens
├── Max Steps: 100,000
├── Batch Size: 8 per device
├── Grad Accumulation: 4
├── Effective Batch: 32
├── Learning Rate: 3e-4
├── Optimizer: AdamW(β=(0.9,0.95), wd=0.1)
├── Scheduler: Cosine with 2% warmup
├── Mixed Precision: bfloat16
└── Gradient Clipping: 1.0

Evaluation:
├── Max Length (T): 1280 (20×W)
├── Checkpoints: 50 (every 2,041 steps)
├── Positions: 1×W, 2×W, 5×W, 10×W, 20×W
└── Method: Chunk PPL (K=S=16)

Dataset:
└── WikiText-103-raw-v1 (full, 2.2M examples)
```

---

## ⏱️ Time Estimates

**Based on previous run (4,250 steps/sec):**

| Steps | Time | Checkpoints |
|-------|------|-------------|
| 15,000 | 42 min | 30 (actual) |
| 25,000 | 70 min | ~12 |
| 50,000 | 2.3 hrs | ~25 |
| 75,000 | 3.5 hrs | ~37 |
| 100,000 | 4.7 hrs | 50 |

**With early stopping:** Likely finishes in 3-5 hours if convergence is reached.

---

## 🎓 Research Questions

### 1. When Does True Convergence Occur?
- At what step does training PPL plateau?
- Does the model keep improving beyond 50k steps?
- What is the final achievable PPL @ 1×W?

### 2. Does SW Advantage Persist?
- In the 15k run, SW won at 10×W (ratio=0.855)
- Is this a **training phase artifact** or **fundamental property**?
- With full convergence, who wins at 10×W?

### 3. Position-Specific Behavior
- Does TTT's advantage at 20×W increase with more training?
- Where is the crossover point (SW better vs TTT better)?
- Can we find an optimal architecture choice per position?

### 4. Training Dynamics
- Is there a phase transition in length generalization ability?
- Does the crossover event (step ~7,755 in short run) repeat?
- How does the learning rate schedule affect generalization?

---

## 📁 Output Structure

```
curriculum_exp_long/
├── metadata.json              # Config + results summary
├── training.log               # Full training log
├── all_summaries.json         # All 50 checkpoint metrics
├── best_model/                # Best checkpoint (min PPL@1xW)
├── final_model/               # Final model state
├── checkpoint_000/            # Initial (untrained)
│   ├── summary.json
│   └── chunk_eval_*.png
├── checkpoint_001/            # Step ~2,041
├── ...
└── checkpoint_049/            # Step ~100,000 or early stop
```

---

## 🔍 Monitoring

### Live Status Check
```bash
python check_curriculum_long_status.py
```

### Continuous Monitoring
```bash
python check_curriculum_long_status.py --loop --interval 180
```

### Training Log
```bash
tail -f curriculum_exp_long/training.log
```

---

## 🎯 Success Criteria

### Training Convergence ✅
- [ ] Training loss plateaus (no improvement for 10 checkpoints)
- [ ] PPL @ 1×W < 500 (ideally < 200)
- [ ] Early stopping triggers naturally

### Length Generalization Analysis ✅
- [ ] All 50 checkpoints evaluated successfully
- [ ] Clear patterns in TTT vs SW comparison
- [ ] Evolution plot shows convergence

### Deliverables ✅
- [ ] Updated `CURRICULUM_RESULTS.md` with full analysis
- [ ] Evolution plot: `curriculum_length_gen_evolution.png`
- [ ] Best model saved and ready for inference
- [ ] Comparison with 15k-step run

---

## 🚀 Next Steps After Completion

1. **Generate Evolution Plot**
   ```bash
   python plot_curriculum_results.py --curriculum_dir curriculum_exp_long
   ```

2. **Compare with Short Run**
   - PPL @ 1×W: 3,154 (15k) vs ??? (100k)
   - PPL @ 10×W: 3,222 (15k) vs ??? (100k)
   - SW/TTT ratio: 0.855 (15k) vs ??? (100k)

3. **Analyze Findings**
   - Update research conclusions
   - Determine if longer training changes architecture recommendations
   - Document when convergence occurred

4. **Model Deployment**
   - Test best model for generation quality
   - Benchmark inference speed
   - Compare with baseline transformers

---

**Status:** Training in progress...  
**Monitor:** `python check_curriculum_long_status.py --loop`  
**ETA:** ~5 hours (October 2, 2025 afternoon)
