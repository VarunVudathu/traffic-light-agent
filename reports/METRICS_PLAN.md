# Evaluation Metrics Plan

## 🎯 Overview

We need metrics that:
1. **Validate hypotheses** - directly test what we claim
2. **Are computable** - can extract from CityFlow
3. **Are meaningful** - traffic engineers care about them
4. **Enable comparison** - distinguish between models

---

## 📊 Recommended Metrics by Hypothesis

### H1: Standard vs Future-Aware DQN

**Primary Hypothesis**: H1-Enhanced should handle traffic surges better

**Recommended Metrics** (Priority Order):

1. ✅ **Recovery Rate** (HIGH PRIORITY)
   - **What**: Time to clear congestion after surge
   - **Why**: Directly tests surge adaptation
   - **Computation**: Steps from surge peak to baseline queue length
   - **Expected**: H1-Enhanced recovers faster than H1-Basic

2. ✅ **Peak Queue Length** (HIGH PRIORITY)
   - **What**: Maximum queue during surge periods
   - **Why**: Shows if model prevents buildup
   - **Computation**: max(queue_lengths) during surge windows
   - **Expected**: H1-Enhanced has lower peaks

3. ✅ **Queue Variance** (MEDIUM PRIORITY)
   - **What**: Standard deviation of queue lengths
   - **Why**: Stability indicator
   - **Computation**: std(queue_lengths_over_time)
   - **Expected**: H1-Enhanced more stable (lower variance)

4. ⚠️ **Average Delay** (ALREADY TRACKING)
   - **What**: Average vehicle travel time
   - **Why**: Overall performance
   - **Computation**: Already in Logger class
   - **Expected**: H1-Enhanced has lower delay

**Skip**:
- ❌ Pressure metric (not core to H1 hypothesis)
- ❌ Gini coefficient (single intersection, fairness not relevant)

---

### H2: MaxPressure Reward Decoupling

**Primary Hypothesis**: MaxPressure reward achieves good throughput with low overhead

**Recommended Metrics** (Priority Order):

1. ✅ **Throughput** (HIGH PRIORITY)
   - **What**: Vehicles completed per episode
   - **Why**: Core performance metric
   - **Computation**: Already in Logger class
   - **Expected**: H2 matches/beats H1-Basic

2. ✅ **Average Pressure** (HIGH PRIORITY)
   - **What**: Mean pressure value during episode
   - **Why**: What H2 is optimizing
   - **Computation**: Track pressure values during run
   - **Expected**: H2 maintains lower pressure

3. ✅ **Computational Overhead** (HIGH PRIORITY)
   - **What**: Inference time per action, training time
   - **Why**: Tests "lower overhead" claim
   - **Computation**: Time measurements
   - **Expected**: H2 comparable to H1-Basic (simple state)

4. ⚠️ **Average Delay** (ALREADY TRACKING)
   - **What**: Average vehicle travel time
   - **Why**: Overall performance
   - **Computation**: Already in Logger
   - **Expected**: H2 competitive with others

**Skip**:
- ❌ Recovery rate (not core to H2 hypothesis)
- ❌ Gini coefficient (single intersection)

---

### H3: Multi-Agent Coordination

**Primary Hypothesis**: Simple phase sharing enables coordination

**Recommended Metrics** (Priority Order):

1. ✅ **Gini Coefficient** (HIGH PRIORITY)
   - **What**: Fairness across intersections
   - **Why**: Tests if coordination balances load
   - **Computation**: Gini(queue_lengths_across_intersections)
   - **Expected**: Shared-Phase has lower Gini (more fair)

2. ✅ **System Throughput** (HIGH PRIORITY)
   - **What**: Total vehicles completed across all intersections
   - **Why**: Overall system performance
   - **Computation**: Sum of throughput across agents
   - **Expected**: Shared-Phase higher than Independent

3. ✅ **Recovery Rate** (HIGH PRIORITY)
   - **What**: Network-wide recovery from congestion
   - **Why**: Tests coordination effectiveness
   - **Computation**: Time for all intersections to clear
   - **Expected**: Shared-Phase recovers faster

4. ✅ **Queue Variance Across Intersections** (MEDIUM PRIORITY)
   - **What**: Variance in queue lengths between intersections
   - **Why**: Load balancing indicator
   - **Computation**: var(mean_queue_per_intersection)
   - **Expected**: Shared-Phase lower variance (balanced)

5. ⚠️ **Average Delay** (ALREADY TRACKING)
   - **What**: Average vehicle travel time
   - **Why**: Overall performance
   - **Computation**: Already in Logger
   - **Expected**: Shared-Phase lower delay

**Skip**:
- ❌ Pressure metric (not core to H3 hypothesis)

---

## 🎯 Final Recommendations

### Metrics to Implement

| Metric | H1 | H2 | H3 | Priority | Complexity |
|--------|----|----|-----|----------|------------|
| **Recovery Rate** | ✅ | ❌ | ✅ | HIGH | Medium |
| **Peak Queue Length** | ✅ | ❌ | ❌ | HIGH | Low |
| **Queue Variance** | ✅ | ❌ | ✅ | MEDIUM | Low |
| **Throughput** | ⚠️ | ✅ | ✅ | HIGH | Already done |
| **Average Pressure** | ❌ | ✅ | ❌ | HIGH | Medium |
| **Computational Overhead** | ❌ | ✅ | ❌ | HIGH | Low |
| **Gini Coefficient** | ❌ | ❌ | ✅ | HIGH | Medium |
| **Average Delay** | ⚠️ | ⚠️ | ⚠️ | MEDIUM | Already done |

**Legend**:
- ✅ Implement for this hypothesis
- ⚠️ Already tracking (Logger class)
- ❌ Skip for this hypothesis

---

## 🛠️ Implementation Plan

### Phase 1: Enhance Logger Class (30 minutes)

Update `models/baselines.py` Logger to track additional metrics:

```python
class EnhancedLogger:
    def __init__(self):
        # Existing
        self.total_delay = 0
        self.total_vehicles = 0
        self.vehicle_passed = set()

        # NEW - Track over time
        self.queue_history = []  # For variance
        self.pressure_history = []  # For H2
        self.peak_queues = []  # For H1
        self.intersection_queues = {}  # For H3 Gini

    def update(self, engine, current_step):
        # Track queue lengths over time
        lane_counts = engine.get_lane_waiting_vehicle_count()
        total_queue = sum(lane_counts.values())
        self.queue_history.append((current_step, total_queue))

        # Track per-intersection for H3
        for iid in engine.get_intersection_ids():
            if iid not in self.intersection_queues:
                self.intersection_queues[iid] = []
            # Get lanes for this intersection
            # ... (implementation)
```

### Phase 2: Add Metric Calculations (45 minutes)

Create `utils/metrics.py`:

```python
import numpy as np

def compute_recovery_rate(queue_history, surge_times):
    """
    Compute recovery rate after surges.

    Args:
        queue_history: List of (step, queue_length) tuples
        surge_times: List of (surge_start, surge_end) tuples

    Returns:
        List of recovery times (one per surge)
    """
    recovery_times = []

    for surge_start, surge_end in surge_times:
        # Find peak during surge
        surge_queues = [q for t, q in queue_history
                        if surge_start <= t <= surge_end]
        peak_queue = max(surge_queues) if surge_queues else 0

        # Find baseline (average before surge)
        baseline_queues = [q for t, q in queue_history
                          if t < surge_start]
        baseline = np.mean(baseline_queues) if baseline_queues else 0
        threshold = baseline + (peak_queue - baseline) * 0.1  # 10% above baseline

        # Find time to recover
        post_surge = [(t, q) for t, q in queue_history if t > surge_end]
        recovery_step = None
        for t, q in post_surge:
            if q <= threshold:
                recovery_step = t
                break

        if recovery_step:
            recovery_time = recovery_step - surge_end
            recovery_times.append(recovery_time)

    return recovery_times

def compute_peak_queue(queue_history, surge_times):
    """Compute peak queue length during surges."""
    peaks = []
    for surge_start, surge_end in surge_times:
        surge_queues = [q for t, q in queue_history
                        if surge_start <= t <= surge_end]
        if surge_queues:
            peaks.append(max(surge_queues))
    return peaks

def compute_queue_variance(queue_history):
    """Compute variance of queue lengths over time."""
    queues = [q for _, q in queue_history]
    return np.var(queues)

def compute_gini_coefficient(values):
    """
    Compute Gini coefficient for fairness.

    Args:
        values: List of values (e.g., mean queue per intersection)

    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)

    # Gini formula
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_values))) / (n * np.sum(sorted_values))
    gini = gini - (n + 1) / n

    return gini

def compute_average_pressure(pressure_history):
    """Compute average pressure over episode."""
    if not pressure_history:
        return 0.0
    return np.mean(pressure_history)

def measure_inference_time(agent, state, num_trials=100):
    """
    Measure average inference time.

    Args:
        agent: Trained agent
        state: Sample state
        num_trials: Number of trials to average

    Returns:
        Average inference time in milliseconds
    """
    import time
    import torch

    times = []
    agent.model.eval()

    with torch.no_grad():
        for _ in range(num_trials):
            start = time.perf_counter()
            q_vals = agent.model(torch.tensor(state).float().unsqueeze(0))
            action = q_vals.argmax().item()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times)
```

### Phase 3: Update Experiment Runners (60 minutes)

Modify `run_h1.py`, create `run_h2.py`, `run_h3.py` to:

1. Instantiate EnhancedLogger
2. Compute new metrics after each run
3. Add to statistical comparison
4. Generate additional plots

Example for H1:

```python
from utils.metrics import compute_recovery_rate, compute_peak_queue, compute_queue_variance

def run_single_experiment_h1(model_type, config_path, seed, episodes=50):
    # ... existing code ...

    # NEW: Track surge times
    surge_times = [(300, 400), (600, 700), (900, 1000)]  # Based on scenario

    logger = EnhancedLogger()

    # ... training loop ...

    # NEW: Compute additional metrics
    recovery_rates = compute_recovery_rate(logger.queue_history, surge_times)
    peak_queues = compute_peak_queue(logger.queue_history, surge_times)
    queue_var = compute_queue_variance(logger.queue_history)

    return {
        'training_rewards': training_rewards,
        'eval_rewards': eval_rewards,
        'metrics': {
            'mean_reward': ...,
            'recovery_rate_mean': np.mean(recovery_rates),
            'recovery_rate_std': np.std(recovery_rates),
            'peak_queue_mean': np.mean(peak_queues),
            'peak_queue_max': max(peak_queues),
            'queue_variance': queue_var,
            # ... existing metrics ...
        }
    }
```

### Phase 4: Enhanced Visualization (30 minutes)

Add plots for new metrics:

```python
def generate_enhanced_plots(results_summary, results_dir):
    # Existing plots ...

    # NEW: Recovery rate comparison
    plt.figure(figsize=(10, 6))
    models = list(results_summary.keys())
    recovery_rates = [results_summary[m]['recovery_rate_mean'] for m in models]
    plt.bar(models, recovery_rates)
    plt.ylabel('Recovery Time (steps)')
    plt.title('Congestion Recovery Rate Comparison')
    plt.savefig(results_dir / 'plots' / 'recovery_rate.png')

    # NEW: Peak queue comparison
    plt.figure()
    peak_queues = [results_summary[m]['peak_queue_mean'] for m in models]
    plt.bar(models, peak_queues)
    plt.ylabel('Peak Queue Length')
    plt.title('Peak Queue During Surges')
    plt.savefig(results_dir / 'plots' / 'peak_queues.png')

    # ... more plots ...
```

---

## 📊 Example Results Table

After implementation, results will look like:

### H1 Comparison

| Model | Recovery Rate (steps) | Peak Queue | Queue Variance | Avg Delay |
|-------|----------------------|------------|----------------|-----------|
| H1-Basic | 145 ± 23 | 38.2 ± 5.1 | 125.3 | 12.4 |
| H1-Enhanced | **98 ± 15** ⬇️ | **29.1 ± 3.2** ⬇️ | **87.6** ⬇️ | **9.8** ⬇️ |
| Fixed-Time | 210 ± 45 | 52.7 ± 8.9 | 203.1 | 18.3 |

**Interpretation**: H1-Enhanced recovers 32% faster, has 24% lower peaks

### H2 Comparison

| Model | Throughput | Avg Pressure | Inference (ms) | Avg Delay |
|-------|------------|--------------|----------------|-----------|
| H1-Basic | 245 ± 12 | -8.2 | 2.3 ± 0.1 | 10.5 |
| H2-MaxPressure | **248 ± 10** ⬆️ | **-6.1** ⬆️ | **2.2 ± 0.1** ⬇️ | **10.1** ⬇️ |

**Interpretation**: H2 matches throughput, lower pressure, same overhead

### H3 Comparison

| Model | Gini Coeff | System Throughput | Recovery Rate | Queue Var |
|-------|------------|-------------------|---------------|-----------|
| H3-Independent | 0.35 ± 0.08 | 412 ± 23 | 178 ± 34 | 145.2 |
| H3-Shared-Phase | **0.18 ± 0.04** ⬇️ | **445 ± 18** ⬆️ | **121 ± 22** ⬇️ | **98.7** ⬇️ |

**Interpretation**: Shared-phase is 49% more fair, 8% higher throughput

---

## ⏱️ Implementation Timeline

| Task | Time | Priority |
|------|------|----------|
| Create utils/metrics.py | 30 min | HIGH |
| Update EnhancedLogger | 30 min | HIGH |
| Modify run_h1.py | 20 min | HIGH |
| Create run_h2.py | 30 min | HIGH |
| Create run_h3.py | 30 min | HIGH |
| Add visualization | 30 min | MEDIUM |
| Test all metrics | 30 min | HIGH |

**Total**: ~3 hours

---

## ✅ Recommended Minimal Set

If you want to **minimize implementation time** while still validating hypotheses:

### For H1 (30 minutes)
- ✅ Recovery Rate (core to hypothesis)
- ✅ Peak Queue Length (easy to compute)
- ⚠️ Average Delay (already have)

### For H2 (20 minutes)
- ✅ Throughput (already have)
- ✅ Computational Overhead (easy to measure)
- ⚠️ Average Delay (already have)

### For H3 (45 minutes)
- ✅ Gini Coefficient (most meaningful for coordination)
- ✅ System Throughput (already have)
- ⚠️ Average Delay (already have)

**Minimal Total**: ~2 hours of implementation

---

## 🎯 My Final Recommendation

### Implement This Subset (Best ROI)

**For all hypotheses**: Use what we already have
- ⚠️ Average Reward (already tracking)
- ⚠️ Throughput (already in Logger)
- ⚠️ Average Delay (already in Logger)

**Add these targeted metrics**:

1. **H1**: Recovery Rate + Peak Queue (45 min)
   - Directly tests surge adaptation claim
   - Easy to visualize and interpret

2. **H2**: Computational Overhead (15 min)
   - Just timing measurements
   - Tests "lower overhead" claim directly

3. **H3**: Gini Coefficient (30 min)
   - Most meaningful for fairness/coordination
   - Distinctive metric for multi-agent

**Total additional work**: ~90 minutes
**Total metrics**: 6-7 per hypothesis
**Value**: Validates all 3 hypotheses convincingly

---

## 📝 Summary

**Use these metrics**:

| Hypothesis | Core Metrics | Why |
|------------|--------------|-----|
| **H1** | Recovery Rate, Peak Queue, Delay | Tests surge adaptation |
| **H2** | Throughput, Overhead, Delay | Tests performance vs cost |
| **H3** | Gini Coefficient, Throughput, Delay | Tests fairness & coordination |

**Skip**:
- ❌ Pressure for H1/H3 (not core to hypothesis)
- ❌ Gini for H1/H2 (single intersection)
- ❌ Queue variance (nice-to-have, not essential)

**Implementation effort**: ~90 minutes for targeted additions

**Payoff**: Clear validation of each hypothesis with meaningful metrics

---

**Ready to implement?** I can start with the metrics.py utility file and enhanced logger!
