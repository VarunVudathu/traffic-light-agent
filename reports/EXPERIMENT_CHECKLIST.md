# Experiment Execution Checklist

A step-by-step guide for running all experiments systematically.

---

## 📋 Pre-Experiment Checklist

### ✅ Environment Setup

- [ ] CityFlow is installed and accessible
- [ ] Python packages installed (`torch`, `numpy`, `scipy`, `matplotlib`, `pandas`)
- [ ] Working directory is `traffic-light-agent/`
- [ ] All scenarios validated (`python scenarios/validate_scenarios.py`)

### ✅ Code Validation

- [ ] Quick test passes (`python test_h1_quick.py`)
- [ ] No syntax errors in model files
- [ ] Import statements work correctly

### ✅ Storage & Resources

- [ ] At least 1GB free disk space for results
- [ ] Sufficient time allocated (60-90 min per hypothesis for full experiments)
- [ ] Computer won't go to sleep during experiments

---

## 🧪 H1 Experiments

### Quick Validation (15 minutes)

- [ ] Run: `python experiments/run_h1_quick.py`
- [ ] Check output shows both experiments
- [ ] Verify statistical output is generated
- [ ] Note any errors or warnings

**Expected Output**:
```
Experiment 1: H1-Basic vs Fixed-Time (Low Variance)
Experiment 2: H1-Enhanced vs H1-Basic (Extreme Surge)
H1 QUICK VALIDATION COMPLETE
```

### Full Experiments (60 minutes)

- [ ] Run: `python experiments/run_h1.py`
- [ ] Monitor progress (check console every 10 min)
- [ ] Wait for completion message
- [ ] Verify outputs created:
  - [ ] `results/h1_results.json` exists
  - [ ] `results/plots/h1_comparison.png` exists
- [ ] Review JSON results: `python -m json.tool results/h1_results.json`
- [ ] View plots: `open results/plots/h1_comparison.png`

**Success Criteria**:
- [ ] All 3 experiments completed (10 runs each)
- [ ] Statistical tests show p-values
- [ ] Plots generated without errors

### H1 Analysis

- [ ] Record which hypothesis won each experiment
- [ ] Note p-values and effect sizes
- [ ] Check if results match expectations:
  - [ ] H1-Basic vs Fixed-Time on low-variance
  - [ ] H1-Enhanced vs H1-Basic on high-variance
  - [ ] H1-Enhanced vs H1-Basic on extreme surge

**Questions to Answer**:
1. Does H1-Basic beat Fixed-Time on low-variance? ________
2. Does H1-Enhanced beat H1-Basic on surge scenarios? ________
3. What is the effect size for key comparisons? ________

---

## 🧪 H2 Experiments

### Preparation

- [ ] Create `experiments/run_h2.py` (based on `run_h1.py` template)
- [ ] Test H2 model works: Quick test with 10 episodes
- [ ] Verify MaxPressure reward is computing correctly

### Quick Validation

- [ ] Run H2 quick test (3 runs, 30 episodes)
- [ ] Compare H2-MaxPressure vs H1-Basic
- [ ] Check reward magnitudes are reasonable

### Full Experiments (60 minutes)

- [ ] Run full H2 experiments (10 runs, 50 episodes)
- [ ] Compare:
  - [ ] H2-MaxPressure vs H1-Basic (same state, different reward)
  - [ ] H2 on different scenarios
- [ ] Measure computational overhead:
  - [ ] Training time per episode
  - [ ] Inference time per action
  - [ ] Memory usage

### H2 Analysis

- [ ] Does MaxPressure reward improve performance? ________
- [ ] Is computational overhead lower than complex state methods? ________
- [ ] Effect size of MaxPressure reward? ________

**Metrics to Record**:
- Training time (H2 vs H1): ________ seconds
- Inference time (H2 vs H1): ________ ms
- Throughput improvement: ________%

---

## 🧪 H3 Experiments

### Preparation

- [ ] Verify roadnet has multiple intersections (or adjust scenario)
- [ ] Create `experiments/run_h3.py`
- [ ] Test both coordination modes work:
  - [ ] H3-Independent
  - [ ] H3-Shared-Phase

### Quick Validation

- [ ] Run H3 quick test (3 runs, 30 episodes each mode)
- [ ] Compare Independent vs Shared-Phase
- [ ] Check coordination is actually happening (observe state sizes)

### Full Experiments (60 minutes)

- [ ] Run H3-Independent (10 runs, 50 episodes)
- [ ] Run H3-Shared-Phase (10 runs, 50 episodes)
- [ ] Test on congestion scenarios:
  - [ ] Moderate surge
  - [ ] Extreme surge

### H3 Analysis

- [ ] Does phase sharing improve coordination? ________
- [ ] Faster congestion recovery with sharing? ________
- [ ] Effect size of coordination? ________

**Metrics to Record**:
- Recovery time (Independent): ________ steps
- Recovery time (Shared-Phase): ________ steps
- System throughput improvement: ________%

---

## 📊 Cross-Hypothesis Analysis

After all experiments complete:

### Performance Comparison

- [ ] Create comparison table of all models
- [ ] Plot all models on same graph
- [ ] Identify best performer overall
- [ ] Identify best performer per scenario

**Comparison Table**:

| Model | Low-Var | High-Var | Surge | Overall |
|-------|---------|----------|-------|---------|
| Fixed-Time | _____ | _____ | _____ | _____ |
| H1-Basic | _____ | _____ | _____ | _____ |
| H1-Enhanced | _____ | _____ | _____ | _____ |
| H2-MaxPressure | _____ | _____ | _____ | _____ |
| H3-Independent | _____ | _____ | _____ | _____ |
| H3-Shared-Phase | _____ | _____ | _____ | _____ |

### Statistical Validation

- [ ] All comparisons have p-values < 0.05 for significance
- [ ] Effect sizes are reasonable (not all negligible)
- [ ] Confidence intervals computed
- [ ] No contradictory results

### Hypothesis Validation

**H1**: Standard vs Future-Aware
- [ ] Hypothesis supported? Yes / No / Partial
- [ ] Key evidence: _________________________________
- [ ] Unexpected findings: _________________________________

**H2**: MaxPressure Reward
- [ ] Hypothesis supported? Yes / No / Partial
- [ ] Key evidence: _________________________________
- [ ] Unexpected findings: _________________________________

**H3**: Multi-Agent Coordination
- [ ] Hypothesis supported? Yes / No / Partial
- [ ] Key evidence: _________________________________
- [ ] Unexpected findings: _________________________________

---

## 📝 Results Documentation

### Create Final Report

- [ ] Compile all results into one document
- [ ] Include all plots
- [ ] Add statistical tables
- [ ] Write interpretation of results
- [ ] Discuss limitations
- [ ] Suggest future work

### Generate Plots

- [ ] Training curves for all models
- [ ] Performance comparison (bar chart with error bars)
- [ ] Scenario-specific comparisons
- [ ] Statistical significance visualization
- [ ] Computational overhead comparison (H2)
- [ ] Recovery rate comparison (H3)

### Export Results

- [ ] Save all JSON results
- [ ] Export CSV for spreadsheet analysis
- [ ] Save high-resolution plots (300 DPI)
- [ ] Archive raw data

---

## 🐛 Troubleshooting During Experiments

### If Training Crashes

1. Check error message
2. Note which episode/run it crashed on
3. Check disk space
4. Check memory usage
5. Resume from checkpoint if possible
6. Reduce batch size or max_steps if memory issue

### If Results Look Wrong

1. Verify scenario files are correct
2. Check if rewards are in expected range
3. Ensure random seeds are different across runs
4. Verify epsilon decay is working
5. Check learning rate isn't too high/low

### If Statistical Tests Fail

1. Check if sample sizes are equal
2. Verify distributions aren't degenerate (all same value)
3. Ensure pairing is correct
4. Check for NaN values

---

## ✅ Completion Checklist

### Before Calling Experiments Complete

- [ ] All 3 hypotheses tested
- [ ] 10 runs per configuration (30 total per hypothesis)
- [ ] All results saved and backed up
- [ ] Plots generated and saved
- [ ] Statistical analysis complete
- [ ] Results reviewed and make sense

### Data Archival

- [ ] Create `results/archive/` directory
- [ ] Copy all results JSON files
- [ ] Copy all plots
- [ ] Create README in archive with experiment date
- [ ] Zip archive: `tar -czf results_YYYY-MM-DD.tar.gz results/`

### Final Report

- [ ] Executive summary written
- [ ] Methods section complete
- [ ] Results section with all plots
- [ ] Discussion of findings
- [ ] Conclusion and future work
- [ ] References cited

---

## 📅 Estimated Timeline

### Conservative Estimates

| Task | Time | Notes |
|------|------|-------|
| H1 Quick Validation | 15 min | Verify code works |
| H1 Full Experiments | 60 min | 10 runs × 50 episodes |
| H1 Analysis | 30 min | Review results |
| H2 Preparation | 30 min | Create run_h2.py |
| H2 Full Experiments | 60 min | 10 runs × 50 episodes |
| H2 Analysis | 30 min | Review + overhead metrics |
| H3 Preparation | 30 min | Create run_h3.py |
| H3 Full Experiments | 60 min | 2 modes × 10 runs |
| H3 Analysis | 30 min | Review + coordination metrics |
| Cross-Analysis | 60 min | Compare all hypotheses |
| Final Report | 120 min | Write comprehensive report |

**Total**: ~8-10 hours (can be spread over multiple days)

---

## 💡 Tips for Success

1. **Start with quick validation** - Don't jump straight to 10 runs
2. **Monitor first run closely** - Catch issues early
3. **Save intermediate results** - Don't lose data to crashes
4. **Document as you go** - Note observations immediately
5. **Take breaks** - Let experiments run while you do other work
6. **Backup frequently** - Copy results to safe location

---

## 🎯 Success Metrics

### Minimum Requirements

- [ ] At least H1 fully completed with statistics
- [ ] All hypotheses tested (even if quick validation only)
- [ ] Statistical analysis performed correctly
- [ ] Results documented and interpretable

### Ideal Completion

- [ ] All 3 hypotheses with full 10-run experiments
- [ ] Comprehensive statistical analysis
- [ ] Publication-ready plots
- [ ] Detailed analysis report
- [ ] Code and data archived

---

## 📞 Help & Resources

- **Code Review**: See `CODE_REVIEW.md`
- **Quick Start**: See `QUICK_START.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Troubleshooting**: See `README.md` section

---

**Last Updated**: 2025-11-29

**Next Action**: Run `python test_h1_quick.py` to verify setup, then start with H1 quick validation!
