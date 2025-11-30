"""
Evaluation metrics for traffic light control experiments.

Provides functions to compute:
- Recovery rate from congestion
- Peak queue lengths
- Gini coefficient for fairness
- Computational overhead
- Queue variance
"""

import numpy as np
import time
import torch


def compute_recovery_rate(queue_history, surge_times, baseline_percentile=0.2):
    """
    Compute recovery rate after traffic surges.

    Recovery is defined as time to return to near-baseline queue levels
    after a surge ends.

    Args:
        queue_history: List of (step, queue_length) tuples
        surge_times: List of (surge_start, surge_end) tuples
        baseline_percentile: Queue threshold as fraction above baseline (default 0.2 = 20%)

    Returns:
        List of recovery times in steps (one per surge)
    """
    if not queue_history or not surge_times:
        return []

    recovery_times = []

    for surge_start, surge_end in surge_times:
        # Find baseline (average before surge)
        baseline_queues = [q for t, q in queue_history if t < surge_start - 50]
        if not baseline_queues:
            baseline_queues = [q for t, q in queue_history if t < surge_start]

        baseline = np.mean(baseline_queues) if baseline_queues else 0

        # Threshold: baseline + 20% of baseline
        threshold = baseline * (1 + baseline_percentile)

        # Find time to recover after surge ends
        post_surge = [(t, q) for t, q in queue_history if t > surge_end]
        recovery_step = None

        for t, q in post_surge:
            if q <= threshold:
                recovery_step = t
                break

        if recovery_step:
            recovery_time = recovery_step - surge_end
            recovery_times.append(recovery_time)
        else:
            # Never recovered, use max
            recovery_times.append(999)  # Indicate failure to recover

    return recovery_times


def compute_peak_queue(queue_history, surge_times):
    """
    Compute peak queue length during surge periods.

    Args:
        queue_history: List of (step, queue_length) tuples
        surge_times: List of (surge_start, surge_end) tuples

    Returns:
        List of peak queue values (one per surge)
    """
    if not queue_history or not surge_times:
        return []

    peaks = []
    for surge_start, surge_end in surge_times:
        surge_queues = [q for t, q in queue_history
                        if surge_start <= t <= surge_end]
        if surge_queues:
            peaks.append(max(surge_queues))
        else:
            peaks.append(0)

    return peaks


def compute_queue_variance(queue_history):
    """
    Compute variance of queue lengths over time.

    Lower variance indicates more stable traffic flow.

    Args:
        queue_history: List of (step, queue_length) tuples

    Returns:
        Variance of queue lengths
    """
    if not queue_history:
        return 0.0

    queues = [q for _, q in queue_history]
    return float(np.var(queues))


def compute_gini_coefficient(values):
    """
    Compute Gini coefficient for fairness/inequality.

    Used to measure how evenly load is distributed across intersections.

    Args:
        values: Array-like of values (e.g., mean queue per intersection)

    Returns:
        Gini coefficient: 0 = perfect equality, 1 = perfect inequality
    """
    if not values or len(values) == 0:
        return 0.0

    values = np.array(values, dtype=float)

    # Handle all zeros case
    if np.sum(values) == 0:
        return 0.0

    # Handle negative values by shifting
    if np.min(values) < 0:
        values = values - np.min(values)

    sorted_values = np.sort(values)
    n = len(values)

    # Gini coefficient formula
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values))
    gini = gini - (n + 1) / n

    return float(gini)


def compute_average_pressure(pressure_history):
    """
    Compute average pressure over episode.

    Args:
        pressure_history: List of pressure values

    Returns:
        Mean pressure value
    """
    if not pressure_history:
        return 0.0
    return float(np.mean(pressure_history))


def measure_inference_time(agent, state, num_trials=100):
    """
    Measure average inference time for action selection.

    Args:
        agent: Trained agent with model
        state: Sample state for inference
        num_trials: Number of trials to average over

    Returns:
        Average inference time in milliseconds
    """
    times = []
    agent.model.eval()

    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            q_vals = agent.model(torch.tensor(state).float().unsqueeze(0))
            _ = q_vals.argmax().item()

        # Actual measurement
        for _ in range(num_trials):
            start = time.perf_counter()
            q_vals = agent.model(torch.tensor(state).float().unsqueeze(0))
            action = q_vals.argmax().item()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    agent.model.train()
    return float(np.mean(times)), float(np.std(times))


def get_surge_times_for_scenario(scenario_name):
    """
    Get surge timing windows for different scenarios.

    Args:
        scenario_name: Name of scenario (e.g., 'moderate_surge', 'extreme_surge')

    Returns:
        List of (surge_start, surge_end) tuples
    """
    surge_configs = {
        'moderate_surge': [
            (300, 400),    # First surge
            (600, 700),    # Second surge
            (900, 1000)    # Third surge
        ],
        'extreme_surge': [
            (350, 500),    # First surge (longer)
            (700, 850),    # Second surge
            # Third surge may be cut off at 1000 steps
        ],
        'high_variance': [],  # No specific surges
        'low_variance': [],   # No specific surges
    }

    return surge_configs.get(scenario_name, [])


def compute_all_metrics(queue_history, scenario_name, logger_metrics=None,
                       pressure_history=None, intersection_queues=None):
    """
    Compute all relevant metrics for a run.

    Args:
        queue_history: List of (step, queue_length) tuples
        scenario_name: Scenario identifier
        logger_metrics: Dict from Logger class (delay, throughput, etc.)
        pressure_history: List of pressure values (for H2)
        intersection_queues: Dict of {intersection_id: queue_values} (for H3)

    Returns:
        Dict of all computed metrics
    """
    metrics = {}

    # Basic metrics from logger
    if logger_metrics:
        metrics.update(logger_metrics)

    # Queue variance (all scenarios)
    metrics['queue_variance'] = compute_queue_variance(queue_history)

    # Surge-specific metrics
    surge_times = get_surge_times_for_scenario(scenario_name)
    if surge_times:
        recovery_rates = compute_recovery_rate(queue_history, surge_times)
        peak_queues = compute_peak_queue(queue_history, surge_times)

        metrics['recovery_rate_mean'] = float(np.mean(recovery_rates)) if recovery_rates else 0.0
        metrics['recovery_rate_std'] = float(np.std(recovery_rates)) if recovery_rates else 0.0
        metrics['peak_queue_mean'] = float(np.mean(peak_queues)) if peak_queues else 0.0
        metrics['peak_queue_max'] = float(max(peak_queues)) if peak_queues else 0.0

    # Pressure metrics (H2)
    if pressure_history:
        metrics['average_pressure'] = compute_average_pressure(pressure_history)

    # Gini coefficient (H3)
    if intersection_queues:
        # Compute mean queue per intersection
        mean_queues = [np.mean(queues) if queues else 0.0
                      for queues in intersection_queues.values()]
        metrics['gini_coefficient'] = compute_gini_coefficient(mean_queues)

    return metrics
