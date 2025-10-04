# W=128 Curriculum Training Experiment

**Status:** 🟢 **RUNNING**  
**Started:** October 2, 2025  
**Training Window:** W = **128 tokens** (2× previous experiments)

---

## 🎯 Experiment Goal

Study how **training window size** affects length generalization by comparing:
- **W=64** (previous experiments)
- **W=128** (this experiment) ← **2× larger training window**

**Key Questions:**
1. Does a larger training window improve length generalization?
2. Do TTT vs SW dynamics change with different window sizes?
3. Is there an optimal window size for extrapolation?

---

## 📊 Configuration

### Model & Training
```python
Model: TTTForCausalLM-125M
├── Training Window (W): 128 tokens  ← 2× larger
├── Eval Max Length (T): 2560 tokens ← 20×W
├── Total Steps: 100,000
├── Eval Checkpoints: 50 (every ~2,041 steps)
├── Batch Size: 8 per device
├── Grad Accumulation: 4
├── Effective Batch: 32
├── Learning Rate: 3e-4
├── Optimizer: AdamW(β=(0.9,0.95), wd=0.1)
├── Scheduler: Cosine with 2% warmup
├── Mixed Precision: bfloat16
└── Early Stopping: Patience=10, min_delta=50 PPL

Dataset: WikiText-103-raw-v1 (full)
```

### Evaluation Positions
We'll measure perplexity at:
- **1×W** = 128 tokens
- **2×W** = 256 tokens
- **5×W** = 640 tokens
- **10×W** = 1,280 tokens
- **20×W** = 2,560 tokens

---

## 🔬 Comparison with W=64 Experiments

| Metric | W=64 (Short) | W=64 (Long) | W=128 (This) |
|--------|--------------|-------------|--------------|
| **Training Window** | 64 | 64 | **128** |
| **Eval Max Length** | 1,280 (20×W) | 1,280 (20×W) | **2,560 (20×W)** |
| **Total Steps** | 15,000 | 100,000 | 100,000 |
| **Eval Checkpoints** | 30 | 50 | 50 |
| **Training Time** | 42 min | ~5-7 hrs | ~5-7 hrs |
| **Final PPL @1×W** | 3,154 | TBD | TBD |
| **Final PPL @10×W (TTT)** | 3,222 | TBD | TBD |
| **Final PPL @10×W (SW)** | 2,755 | TBD | TBD |
| **SW/TTT Ratio @10×W** | 0.855 | TBD | TBD |

---

## 🤔 Research Hypotheses

### Hypothesis 1: Larger Window → Better Absolute Performance
- **Prediction:** W=128 will achieve lower PPL than W=64 at all positions
- **Reasoning:** More training context = better language modeling
- **Test:** Compare PPL @ 1×W between experiments

### Hypothesis 2: Larger Window → Better Extrapolation
- **Prediction:** W=128 will show better relative generalization (e.g., PPL@10×W / PPL@1×W ratio)
- **Reasoning:** Longer context during training teaches better long-range dependencies
- **Test:** Compare extrapolation ratios

### Hypothesis 3: SW Advantage Diminishes with Larger Window
- **Prediction:** At W=128, TTT might outperform SW at 10×W
- **Reasoning:** SW's recency bias less helpful when window is already large
- **Test:** Compare SW/TTT ratio @ 10×W between W=64 and W=128

### Hypothesis 4: Training is Slower with Larger Window
- **Prediction:** W=128 will take longer per step due to increased compute
- **Reasoning:** TTT layers process longer sequences
- **Test:** Compare steps/sec between experiments

---

## 📈 Expected Results Scenarios

### Scenario A: "Bigger is Always Better"
```
W=128 outperforms W=64 at all positions
→ Recommendation: Use largest feasible window
```

### Scenario B: "Diminishing Returns"
```
W=128 better at 1×W, similar at 10×W, worse at 20×W
→ Recommendation: W=64 is sweet spot for extrapolation
```

### Scenario C: "Different Optimal Windows"
```
W=64 better for extreme extrapolation (>15×W)
W=128 better for moderate range (5-10×W)
→ Recommendation: Choose based on target use case
```

---

## 🔍 Monitoring

### Quick Status
```bash
python check_curriculum_long_status.py --curriculum_dir curriculum_exp_w128
```

### Continuous Monitoring
```bash
python check_curriculum_long_status.py --curriculum_dir curriculum_exp_w128 --loop --interval 180
```

### Training Log
```bash
tail -f curriculum_exp_w128/training.log
```

---

## 📊 Analysis Plan (After Completion)

### 1. Generate Evolution Plots
```bash
python plot_curriculum_results.py --curriculum_dir curriculum_exp_w128
```

### 2. Compare with W=64
Create comparison plots showing:
- PPL @ 1×W over training (both windows)
- PPL @ 10×W over training (both windows)
- Final extrapolation curves (W=64 vs W=128)
- SW/TTT ratio evolution

### 3. Key Metrics to Extract
- **Convergence Speed:** At what step does each reach plateau?
- **Final Performance:** PPL @ 1×W and 10×W for both
- **Extrapolation Quality:** Ratio of PPL@10×W / PPL@1×W
- **Architecture Preference:** Where does TTT win vs SW for each window?

### 4. Statistical Tests
- Is improvement from W=64 to W=128 significant?
- Does crossover point change with window size?
- Compute correlation between window size and generalization

---

## 📁 Output Structure

```
curriculum_exp_w128/
├── metadata.json              # Config + results
├── training.log               # Full log
├── all_summaries.json         # 50 checkpoint metrics
├── best_model/                # Best checkpoint
├── final_model/               # Final model
├── checkpoint_000/            # Untrained
│   ├── summary.json
│   └── chunk_eval_W128_T2560_*.png
├── checkpoint_001/            # Step ~2,041
├── ...
├── checkpoint_049/            # Step ~100k or early stop
└── curriculum_length_gen_evolution.png  # Main result plot
```

---

## ⏱️ Timeline

| Time | Event |
|------|-------|
| **Now** | Training started (data loading phase) |
| **+10 min** | Initial eval (checkpoint 0 - untrained) |
| **+1 hr** | Checkpoint ~12 (step ~24k) |
| **+2.5 hrs** | Checkpoint ~25 (step ~50k) |
| **+5 hrs** | Checkpoint ~50 or early stop |
| **+5.5 hrs** | Generate plots and analysis |

**Expected completion:** October 2, 2025 afternoon/evening

---

## 🎓 Learning Objectives

By the end of this experiment, we should be able to answer:

1. ✅ **Does window size matter for length generalization?**
   - Quantify the effect of 2× larger window
   
2. ✅ **Is there a sweet spot for training window size?**
   - Balance between performance and extrapolation ability
   
3. ✅ **How does architecture choice depend on window size?**
   - When to use TTT vs SW based on window size
   
4. ✅ **What are the computational tradeoffs?**
   - Training speed vs performance improvements

---

## 🚀 Related Experiments

### Completed
- ✅ **W=64, 15k steps** (`curriculum_exp/`)
  - Result: PPL@1×W=3,154, SW wins @10×W (ratio=0.855)
  - Issue: Undertrained (still improving at end)

### In Progress
- 🟢 **W=64, 100k steps** (`curriculum_exp_long/`)
  - Goal: Reach full convergence with W=64
  - Status: Data loading phase
  
- 🟢 **W=128, 100k steps** (`curriculum_exp_w128/`) ← **This experiment**
  - Goal: Study effect of larger training window
  - Status: Data loading phase

### Future (Potential)
- ⏳ **W=256, 100k steps** - Even larger window
- ⏳ **W=64 vs W=128 direct comparison** - Same random seed, different windows
- ⏳ **Adaptive window size** - Dynamic window during training

---

**Status:** Training launched successfully  
**Monitor:** `python check_curriculum_long_status.py --curriculum_dir curriculum_exp_w128 --loop`  
**Next Update:** When checkpoint 1 completes (~30-40 min)

---

**Last Updated:** October 2, 2025  
**Experiment ID:** `curriculum_exp_w128`
