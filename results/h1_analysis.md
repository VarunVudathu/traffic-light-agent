# H1 Experimental Results Analysis

## Executive Summary

Hypothesis H1 tested whether adding temporal features (future-aware DQN) would improve traffic light control performance compared to basic DQN, particularly during traffic surges. **The hypothesis was rejected** - temporal features actually degraded performance rather than improving it.

### Key Findings

1. **Basic DQN undertrained**: Both H1-Basic and H1-Enhanced performed significantly worse than the Fixed-Time baseline
2. **Temporal features hurt performance**: H1-Enhanced showed worse surge recovery (0% recovery rate) compared to H1-Basic (30% recovery rate)
3. **Insufficient training duration**: 50 episodes with epsilon at 0.778 indicates models never fully converged

---

## Experimental Design

### Configurations Tested

- **H1-Basic**: Standard DQN with state = [queue lengths] (N dimensions)
- **H1-Enhanced**: DQN with temporal features = [queue lengths, phase duration, queue derivatives, phase history] (2N+1+M dimensions)
- **Fixed-Time**: Deterministic baseline with 30-second fixed cycles

### Test Scenarios

1. **Low Variance** (Experiment 1): Stable traffic with minimal fluctuations
2. **High Variance** (Experiment 2a): Random traffic patterns with moderate variability
3. **Extreme Surge** (Experiment 2b): Multiple traffic surge events at specific time windows

### Methodology

- **10 independent runs** per configuration with different random seeds
- **50 training episodes** per run
- **1000 simulation steps** per evaluation episode
- **Paired t-tests** for statistical comparison (α = 0.05)
- **Cohen's d** for effect size measurement

---

## Results

### Experiment 1: H1-Basic vs Fixed-Time (Low Variance)

**Research Question**: Can basic DQN match fixed-time performance on stable traffic?

| Metric | H1-Basic | Fixed-Time | Statistical Test |
|--------|----------|------------|------------------|
| Mean Reward | -7032.94 ± 1009.21 | -4.67 ± 0.00 | t = -20.89, **p < 0.001*** |
| Cohen's d | -6.607 | - | **Large effect** |
| Winner | ❌ | ✅ | Fixed-Time significantly better |
| Queue Variance | 39,892.8 | 11,655.9 | 3.4× higher variance |
| Throughput | 0 vehicles | 0 vehicles | Both = 0 |

**Interpretation**:
- DQN failed to learn effective policies after 50 episodes
- Highly variable performance across runs (std = 1009.21)
- Fixed-time controller's deterministic behavior provided superior stability
- Zero throughput indicates evaluation window too short for vehicles to complete trips

### Experiment 2a: H1-Enhanced vs H1-Basic (High Variance)

**Research Question**: Do temporal features help in unpredictable traffic?

| Metric | H1-Enhanced | H1-Basic | Statistical Test |
|--------|-------------|----------|------------------|
| Mean Reward | -7794.47 ± 1322.44 | -7315.59 ± 619.57 | t = -0.86, **p = 0.410** |
| Cohen's d | -0.273 | - | Small effect |
| Winner | - | - | **No significant difference** |

**Interpretation**:
- No evidence that temporal features improve performance on high-variance traffic
- H1-Enhanced showed higher variance (1322.44 vs 619.57), suggesting less stable learning
- Larger state space (2N+1+M dimensions) may require more training samples

### Experiment 2b: H1-Enhanced vs H1-Basic (Extreme Surge)

**Research Question**: Do temporal features improve surge recovery?

| Metric | H1-Enhanced | H1-Basic | Statistical Test |
|--------|-------------|----------|------------------|
| Mean Reward | -8454.26 ± 1241.64 | -7453.82 ± 578.07 | t = -2.61, **p = 0.028*** |
| Cohen's d | -0.826 | - | **Large effect** |
| Winner | ❌ | ✅ | H1-Basic significantly better |

**Enhanced Surge Metrics**:

| Metric | H1-Enhanced | H1-Basic | Comparison |
|--------|-------------|----------|------------|
| Recovery Rate | **0/10 runs** (0%) | **3/10 runs** (30%) | H1-Basic 30% better |
| Mean Recovery Time | 999 steps (timeout) | 813 steps | H1-Enhanced never recovered |
| Peak Queue (mean) | 507.4 ± 79.8 | 509.5 ± 56.9 | Similar peaks |
| Peak Queue (max) | 709.5 vehicles | 631.0 vehicles | H1-Enhanced 12% worse |
| Queue Variance | 21,236.5 | 16,996.8 | H1-Enhanced 25% higher |

**Critical Finding**: H1-Enhanced **never successfully recovered** from traffic surges in any of the 10 runs, while H1-Basic recovered in 30% of cases. This directly contradicts the hypothesis that temporal features improve surge adaptation.

**Recovery Time Distribution**:
- H1-Basic: [521, 999, 999, 999, 519, 999, 999, 999, 500, 999] steps
  - Best case: 500 steps post-surge
  - Success rate: 30% (3/10)
- H1-Enhanced: [999, 999, 999, 999, 999, 999, 999, 999, 999, 999] steps
  - All runs timed out
  - Success rate: 0% (0/10)

---

## Analysis

### Why Did H1-Enhanced Underperform?

#### 1. **Curse of Dimensionality**
- **State space expansion**: H1-Enhanced has 2N+1+M dimensions vs N for H1-Basic
- **Sample efficiency**: Larger state spaces require exponentially more samples to explore
- **Training duration**: 50 episodes insufficient for high-dimensional space
- **Evidence**: Higher variance in H1-Enhanced results (1322.44 vs 619.57)

#### 2. **Feature Engineering Issues**
- **Queue derivatives**: May introduce noise without proper smoothing
- **Phase history**: One-hot encoding adds M dimensions but may not capture useful patterns
- **Phase duration**: Normalized by max_phase_time, but relationship to optimal control may be non-linear

#### 3. **Credit Assignment Problem**
- **Temporal features**: Add historical information but complicate credit assignment
- **Delayed rewards**: DQN struggles to attribute rewards to distant past states
- **Surge recovery**: Requires learning long-horizon dependencies that 50 episodes can't capture

#### 4. **Exploration-Exploitation Imbalance**
- **Final epsilon**: 0.778 after training (decay rate = 0.995)
- **Insufficient exploitation**: Models still exploring 78% of the time at evaluation
- **Random actions**: Temporal features provide no benefit when actions are mostly random

### Why Did Basic DQN Fail vs Fixed-Time?

#### 1. **Undertraining**
- **Episodes**: 50 is very low for DQN convergence (typical: 500-5000)
- **Epsilon decay**: Starting at 0.951, only decayed to 0.778 after 50 episodes
- **Buffer size**: 50,000 with only ~50,000 samples collected (50 episodes × 1000 steps)

#### 2. **Reward Signal Quality**
- **Negative queue length**: Simple but may not capture all objectives
- **No shaping**: No intermediate rewards for good behavior
- **Sparse feedback**: Only receives signal when queues change

#### 3. **Fixed-Time Advantages**
- **Domain-tuned**: 30-second cycles likely optimized for this traffic pattern
- **Zero variance**: Deterministic behavior provides perfect consistency
- **Provably stable**: Guarantees fairness and avoids starvation

---

## Recommendations

### For H1 Hypothesis

#### Reject H1 as Currently Formulated
The hypothesis that temporal features improve DQN performance is **not supported** by the evidence. Temporal features:
- Did not improve high-variance traffic handling (p = 0.410)
- **Significantly degraded** surge recovery performance (p = 0.028, 0% vs 30% recovery)
- Increased training instability (higher variance across runs)

#### Potential Salvage Paths

If continuing with H1:

1. **Increase training duration dramatically**
   - Target: 500-1000 episodes minimum
   - Adjust epsilon decay to reach ~0.1 by end of training
   - Implement early stopping based on validation performance

2. **Simplify temporal features**
   - Test each feature individually (ablation study)
   - Remove queue derivatives (likely adding noise)
   - Consider learned temporal embeddings instead of hand-crafted features

3. **Add reward shaping**
   - Penalize high queue variance
   - Reward smooth phase transitions
   - Include fairness terms across lanes

4. **Improve feature engineering**
   - Normalize features properly (z-score instead of manual scaling)
   - Add domain knowledge (e.g., upstream/downstream queue relationships)
   - Use moving averages for queue derivatives

5. **Implement proper hyperparameter tuning**
   - Grid search over: learning rate, batch size, buffer size, network architecture
   - Use validation set to prevent overfitting
   - Track training curves to diagnose issues

### For Overall Project

1. **Establish better baselines**
   - Test fixed-time with multiple cycle lengths
   - Implement MaxPressure as heuristic baseline (useful for H2)
   - Consider Actuated control (extends green based on detection)

2. **Validate experimental setup**
   - Verify throughput calculation (currently showing 0)
   - Confirm surge timing windows align with scenario files
   - Add convergence checks during training

3. **Consider alternative approaches**
   - **H2 (MaxPressure reward)**: May help by providing better gradient signal
   - **H3 (Multi-agent)**: Simpler per-agent state spaces may train faster
   - Actor-Critic methods (PPO, A3C): Better for continuous control
   - Imitation learning: Bootstrap from fixed-time or MaxPressure

---

## Statistical Validity

### Test Assumptions
- **Independence**: 10 runs with different seeds ✅
- **Normality**: Sample size n=10 borderline for t-test, but acceptable for pilot study
- **Paired design**: Same scenarios across configurations ✅

### Effect Sizes
- Experiment 1: d = -6.607 (**very large effect**)
- Experiment 2a: d = -0.273 (small effect, not significant)
- Experiment 2b: d = -0.826 (**large effect**)

### Statistical Power
With n=10, α=0.05:
- Power to detect large effects (d > 0.8): ~80%
- Power to detect medium effects (d > 0.5): ~50%
- Power to detect small effects (d > 0.2): ~15%

**Conclusion**: Adequate power for detecting large effects (Exp 1, 2b) but limited for medium/small effects (Exp 2a).

---

## Conclusion

**H1 Hypothesis Status: REJECTED**

The experimental evidence does not support the hypothesis that temporal features improve DQN performance for traffic light control. In fact, temporal features significantly **degraded** performance during extreme surge scenarios:

- **0% surge recovery rate** vs 30% for basic DQN
- **25% higher queue variance** indicating less stable control
- **No improvement** in high-variance scenarios

The more fundamental finding is that **50 episodes is insufficient** for DQN convergence in this domain. Both H1-Basic and H1-Enhanced failed to match the simple Fixed-Time baseline.

**Next Steps**:
1. Re-run experiments with 500+ episodes if pursuing DQN
2. Consider alternative RL algorithms (PPO, SAC)
3. Proceed with H2/H3 experiments which may provide better foundations
4. Implement proper baseline comparisons (MaxPressure, Actuated)

---

## Appendix: Detailed Results

### Raw Performance Data

**Experiment 1 - H1-Basic Performance by Run**:
```
Run  1: -7626.55
Run  2: -8438.73
Run  3: -7238.90
Run  4: -8648.76
Run  5: -5830.76
Run  6: -5868.42
Run  7: -5880.31
Run  8: -6113.87
Run  9: -7479.64
Run 10: -7203.48
Mean: -7032.94 ± 1009.21
```

**Experiment 2b - Recovery Behavior**:

H1-Enhanced - Complete Failure Pattern:
- All 10 runs timed out at 999 steps
- Mean peak queue: 507.4 vehicles
- Max peak queue: 709.5 vehicles (Run 7)
- Mean queue variance: 21,236.5

H1-Basic - Partial Success Pattern:
- 3/10 runs recovered (Runs 1, 5, 10)
- Recovery times: 521, 519, 500 steps
- Mean peak queue: 509.5 vehicles
- Max peak queue: 631.0 vehicles (Run 6)
- Mean queue variance: 16,996.8

**Queue Variance Analysis**:

Experiment 1 (Low Variance):
- H1-Basic: 39,892.8 (3.4× baseline)
- Fixed-Time: 11,655.9 (baseline)

Experiment 2b (Extreme Surge):
- H1-Enhanced: 21,236.5 (1.25× H1-Basic)
- H1-Basic: 16,996.8

Lower queue variance indicates more stable, predictable traffic flow.

---

**Generated**: 2025-11-30
**Experiments**: H1 Full Suite (30 runs total)
**Training**: 50 episodes per run, 1000 steps per episode
**Evaluation**: 1000 steps per run with metrics collection
