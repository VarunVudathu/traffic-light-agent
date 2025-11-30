# Implementation Plan: Three DQN Hypotheses

## Overview
We need to implement and statistically compare three different DQN architectures for traffic light control.

---

## H1: Single-Agent DQN - Standard vs Future-Aware

### Hypothesis Statement
> A Single-Agent DQN using standard queue metrics will outperform Fixed-Time controllers in low-variance traffic, but will fail to adapt to high-variance surges due to the lack of 'future-aware' state features.

### Implementation Requirements

#### 1.1 Two Variants
- **H1-Basic**: Current implementation (queue snapshots only)
- **H1-Enhanced**: Add temporal features:
  - Current phase duration (how long current light has been active)
  - Queue length derivative (rate of change)
  - Phase history (last N phase switches)

#### 1.2 Traffic Scenarios
- **Low-variance**: Constant arrival rate (interval=5.0s, σ=0.5s)
- **High-variance surge**: Poisson arrivals with periodic surges
  - Base: interval=5.0s
  - Surge periods: interval=1.0s for 100 steps every 300 steps

#### 1.3 Metrics to Collect
- Average delay
- Throughput (vehicles completed)
- Average queue length
- Standard deviation of queue length
- Response time to surges (time to clear surge queue)

#### 1.4 Statistical Tests
- **Paired t-test**: H1-Basic vs Fixed-Time on low-variance
- **Paired t-test**: H1-Basic vs H1-Enhanced on high-variance
- Run each configuration **10 times** with different seeds
- Report: mean ± std, p-value, effect size (Cohen's d)

---

## H2: MaxPressure Reward with Simplified State

### Hypothesis Statement
> Decoupling the reward function (MaxPressure) from state (simplified queue snapshots) will achieve PressLight-level throughput with lower computational overhead.

### Implementation Requirements

#### 2.1 MaxPressure Definition
```
Pressure = Σ(incoming_lane_vehicles) - Σ(outgoing_lane_vehicles)
Reward = max_pressure_across_phases
```

#### 2.2 Two Variants
- **H2-MaxPressure**: MaxPressure reward + simple state (queue counts)
- **H2-PressLight**: MaxPressure reward + pressure-based state (for comparison)

#### 2.3 State Representation
- **Simple**: Just queue counts (current)
- **Pressure-based**: Pressure values per phase

#### 2.4 Computational Overhead Metrics
- Training time per episode
- Inference time per action
- Model size (parameters)
- Memory usage

#### 2.5 Performance Metrics
- Throughput (primary)
- Average travel time
- Computational cost (FLOPs/action)

#### 2.6 Statistical Tests
- **Wilcoxon signed-rank test**: H2-MaxPressure vs H2-PressLight on throughput
- **Computational overhead comparison**: t-test on inference time
- 10 runs per configuration

---

## H3: Multi-Agent - Simple Shared-Phase vs Complex GAT

### Hypothesis Statement
> Explicit sharing of immediate neighbor phases is sufficient for coordination in grid topologies. Simple Shared-Phase DQN will match complex GAT-based models during congestion peaks.

### Implementation Requirements

#### 3.1 Three Variants
- **H3-Independent**: Each intersection has own DQN, no sharing
- **H3-Shared-Phase**: DQN with neighbor phase information
  - State: own_queues + neighbor_current_phases
- **H3-GAT** (optional): Graph Attention Network (if time permits)

#### 3.2 Network Topology
- Use 2x2 or 3x3 grid of intersections
- Each agent observes immediate neighbors (4-connected)

#### 3.3 Coordination Metrics
- **Recovery rate**: Time to clear congestion after peak
- **System throughput**: Total vehicles across all intersections
- **Phase synchronization**: Measure "green wave" coordination
- **Queue variance**: Distribution across intersections

#### 3.4 Test Scenarios
- **Congestion peak**: Sudden traffic surge at specific intersections
- **Wave propagation**: Traffic surge moves across grid

#### 3.5 Statistical Tests
- **ANOVA**: Compare all three variants on recovery rate
- Post-hoc: Tukey HSD for pairwise comparisons
- 10 runs per configuration with different surge patterns

---

## Statistical Comparison Framework

### Overall Design
```
For each hypothesis:
  For each variant:
    For seed in [1, 2, ..., 10]:
      Train model
      Evaluate on test scenarios (5 episodes each)
      Record metrics
    Compute: mean, std, 95% CI
  Run statistical tests
  Generate comparison plots with error bars
```

### Statistical Tests to Implement

1. **Paired t-test** (parametric)
   - When: Comparing two variants on same scenarios
   - Requirements: Normal distribution of differences
   - Output: t-statistic, p-value, Cohen's d

2. **Wilcoxon signed-rank test** (non-parametric)
   - When: Non-normal distributions
   - Backup for t-test

3. **ANOVA + Tukey HSD** (for H3)
   - When: Comparing 3+ variants
   - Output: F-statistic, p-value, pairwise comparisons

4. **Effect Size Metrics**
   - Cohen's d: (mean1 - mean2) / pooled_std
   - Interpretation: small (0.2), medium (0.5), large (0.8)

### Visualization Requirements

1. **Performance Comparison Plots**
   - Bar charts with error bars (mean ± std)
   - Box plots for distribution visualization
   - Statistical significance markers (*, **, ***)

2. **Training Curves**
   - Shaded regions for confidence intervals
   - Smoothed curves across multiple runs

3. **Scenario-Specific Analysis**
   - Low-variance vs high-variance (H1)
   - Computational cost vs performance (H2)
   - Recovery time analysis (H3)

---

## Implementation Order

### Phase 1: Traffic Scenario Generation
1. Create variance-controlled flow generators
2. Create surge scenario generators
3. Validate scenarios work with CityFlow

### Phase 2: H1 Implementation
1. Implement H1-Enhanced (temporal features)
2. Create low/high variance test scenarios
3. Run experiments + statistical analysis
4. Generate H1 results report

### Phase 3: H2 Implementation
1. Implement MaxPressure reward calculation
2. Implement H2-MaxPressure variant
3. Add computational profiling
4. Run experiments + statistical analysis
5. Generate H2 results report

### Phase 4: H3 Implementation
1. Extend environment to multi-intersection
2. Implement neighbor observation
3. Implement H3-Shared-Phase variant
4. Create congestion peak scenarios
5. Run experiments + statistical analysis
6. Generate H3 results report

### Phase 5: Final Comparison
1. Cross-hypothesis comparison
2. Master results table
3. Final report with all statistical tests

---

## File Structure

```
traffic-light-agent/
├── models/
│   ├── h1_basic.py           # Current implementation
│   ├── h1_enhanced.py         # With temporal features
│   ├── h2_maxpressure.py      # MaxPressure reward
│   ├── h3_shared_phase.py     # Multi-agent coordination
│   └── baselines.py           # Fixed-time, PressLight
├── scenarios/
│   ├── generate_variance.py   # Low/high variance flows
│   ├── generate_surge.py      # Surge scenarios
│   └── configs/               # All scenario configs
├── experiments/
│   ├── run_h1.py             # H1 experiments
│   ├── run_h2.py             # H2 experiments
│   ├── run_h3.py             # H3 experiments
│   └── statistical_analysis.py # All statistical tests
├── results/
│   ├── h1_results.csv
│   ├── h2_results.csv
│   ├── h3_results.csv
│   └── plots/
└── reports/
    ├── h1_report.md
    ├── h2_report.md
    ├── h3_report.md
    └── final_report.md
```

---

## Success Criteria

### H1
- [ ] H1-Basic beats Fixed-Time on low-variance (p < 0.05)
- [ ] H1-Enhanced beats H1-Basic on high-variance (p < 0.05)
- [ ] Clear degradation of H1-Basic on surge scenarios

### H2
- [ ] H2-MaxPressure matches H2-PressLight throughput (no sig. diff)
- [ ] H2-MaxPressure has lower computational overhead (p < 0.05)
- [ ] Effect size: medium or larger (d > 0.5)

### H3
- [ ] H3-Shared-Phase matches H3-GAT on recovery rate (no sig. diff)
- [ ] Both beat H3-Independent (p < 0.05)
- [ ] Coordination metrics show synchronization

---

## Timeline Estimate

- Phase 1 (Scenarios): 2-3 hours
- Phase 2 (H1): 4-5 hours
- Phase 3 (H2): 4-5 hours
- Phase 4 (H3): 6-8 hours
- Phase 5 (Final): 2-3 hours

**Total**: ~20-25 hours of implementation + training time
