"""
H2 Experiment Runner with Statistical Analysis.

Tests H2 hypothesis:
"Decoupling reward from queue length to MaxPressure will improve learning efficiency
and convergence compared to standard queue-based rewards (H1-Basic)."

Experiments:
1. H2-MaxPressure vs H1-Basic on low-variance traffic
2. H2-MaxPressure vs MaxPressure-Baseline on low-variance traffic
3. H2-MaxPressure vs H1-Basic on high-variance traffic
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.h1_basic import CityFlowEnv, H1BasicAgent
from models.h2_maxpressure import CityFlowEnvMaxPressure, H2MaxPressureAgent
from models.baselines import FixedTimeBaseline, MaxPressureBaseline
from utils.logger import EnhancedLogger
from utils.metrics import compute_all_metrics


def compute_metrics(rewards, max_steps):
    """Compute performance metrics from reward history."""
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'total_reward': np.sum(rewards),
        'final_100_mean': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    }


def evaluate_with_logging(agent, env, scenario_name, max_steps=1000, use_maxpressure_reward=False):
    """
    Evaluate agent with enhanced logging.

    Args:
        agent: Trained agent (or None for baseline)
        env: Environment or baseline controller
        scenario_name: Name of scenario
        max_steps: Max simulation steps
        use_maxpressure_reward: Whether to use MaxPressure reward (for H2)

    Returns:
        Dict with enhanced metrics
    """
    logger = EnhancedLogger()
    rewards = []

    # For DQN agents
    if hasattr(agent, 'model'):
        agent.model.eval()
        state = env.reset()

        for step in range(max_steps):
            with torch.no_grad():
                q_vals = agent.model(torch.tensor(state).float().unsqueeze(0))
                action = q_vals.argmax().item()

            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            logger.update(env.eng, current_step=step)

            state = next_state
            if done:
                break

        agent.model.train()

    # For baseline controllers
    else:
        baseline = env
        baseline.eng.reset()

        for step in range(max_steps):
            # Update traffic lights
            for iid, state in baseline.intersection_state.items():
                phases = baseline.eng.get_intersection_phase(iid)
                if not phases:
                    continue

                state['time_in_phase'] += 1

                # Different logic for MaxPressure vs Fixed-Time
                if isinstance(baseline, MaxPressureBaseline):
                    if state['time_in_phase'] >= baseline.min_phase_duration:
                        from utils.maxpressure import estimate_lane_pressure_by_phase
                        pressures = estimate_lane_pressure_by_phase(
                            baseline.eng, iid, state['num_phases']
                        )
                        best_phase = int(np.argmax(pressures))
                        if best_phase != state['current_phase']:
                            state['current_phase'] = best_phase
                            state['time_in_phase'] = 0
                else:
                    # Fixed-time
                    if state['time_in_phase'] >= baseline.phase_duration:
                        state['current_phase'] = (state['current_phase'] + 1) % len(phases)
                        state['time_in_phase'] = 0

                baseline.eng.set_tl_phase(iid, state['current_phase'])

            baseline.eng.next_step()

            # Compute reward
            if use_maxpressure_reward:
                from utils.maxpressure import compute_maxpressure_reward
                reward = compute_maxpressure_reward(
                    baseline.eng,
                    baseline.intersection_ids[0] if baseline.intersection_ids else None,
                    baseline.intersection_state[baseline.intersection_ids[0]]['current_phase'] if baseline.intersection_ids else 0
                )
            else:
                waiting_counts = list(baseline.eng.get_lane_waiting_vehicle_count().values())
                reward = -np.mean(waiting_counts) if waiting_counts else 0.0

            rewards.append(reward)
            logger.update(baseline.eng, current_step=step)

    enhanced_metrics = logger.get_enhanced_metrics(scenario_name=scenario_name)
    enhanced_metrics['rewards'] = rewards
    enhanced_metrics['mean_reward'] = np.mean(rewards)
    return enhanced_metrics


def run_single_experiment(model_type, config_path, seed, scenario_name='', episodes=50):
    """
    Run a single experiment.

    Args:
        model_type: 'h1_basic', 'h2_maxpressure', 'fixed_time', or 'maxpressure_baseline'
        config_path: Path to traffic config
        seed: Random seed
        scenario_name: Scenario identifier
        episodes: Number of training episodes

    Returns:
        Dict with results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Run with seed={seed}...")

    if model_type == 'h1_basic':
        env = CityFlowEnv(config_path, frame_skip=1, max_steps=1000)
        agent = H1BasicAgent(env, learning_rate=1e-3)
        training_rewards = agent.train(
            episodes=episodes,
            epsilon_start=0.951,
            epsilon_end=0.1,
            decay=0.995
        )
        env_fresh = CityFlowEnv(config_path, frame_skip=1, max_steps=1000)
        enhanced_metrics = evaluate_with_logging(agent, env_fresh, scenario_name, max_steps=1000)

    elif model_type == 'h2_maxpressure':
        env = CityFlowEnvMaxPressure(config_path, frame_skip=1, max_steps=1000)
        agent = H2MaxPressureAgent(env, learning_rate=1e-3)
        training_rewards = agent.train(
            episodes=episodes,
            epsilon_start=0.951,
            epsilon_end=0.1,
            decay=0.995
        )
        env_fresh = CityFlowEnvMaxPressure(config_path, frame_skip=1, max_steps=1000)
        enhanced_metrics = evaluate_with_logging(agent, env_fresh, scenario_name, max_steps=1000, use_maxpressure_reward=True)

    elif model_type == 'maxpressure_baseline':
        baseline = MaxPressureBaseline(config_path, min_phase_duration=5, max_steps=1000)
        training_rewards = []
        enhanced_metrics = evaluate_with_logging(None, baseline, scenario_name, max_steps=1000, use_maxpressure_reward=True)

    elif model_type == 'fixed_time':
        baseline = FixedTimeBaseline(config_path, phase_duration=30, max_steps=1000)
        training_rewards = []
        enhanced_metrics = evaluate_with_logging(None, baseline, scenario_name, max_steps=1000)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    eval_rewards = enhanced_metrics['rewards']
    metrics = compute_metrics(eval_rewards, 1000)

    print(f"  Mean reward: {metrics['mean_reward']:.2f}")

    return {
        'training_rewards': training_rewards,
        'eval_rewards': eval_rewards,
        'metrics': metrics,
        'enhanced_metrics': enhanced_metrics
    }


def run_comparison(group1_type, group2_type, config_path, scenario_name='', num_runs=10, episodes=50):
    """Run comparison experiment between two model types."""
    print(f"\nRunning: {group1_type} on {scenario_name}")
    print("=" * 70)

    group1_results = []
    for i in range(num_runs):
        seed = 42 + i
        print(f"Run {i+1}/{num_runs} (seed={seed})...")
        result = run_single_experiment(group1_type, config_path, seed, scenario_name, episodes)
        group1_results.append(result)

    print(f"\nRunning: {group2_type} on {scenario_name}")
    print("=" * 70)

    group2_results = []
    for i in range(num_runs):
        seed = 42 + i
        print(f"Run {i+1}/{num_runs} (seed={seed})...")
        result = run_single_experiment(group2_type, config_path, seed, scenario_name, episodes)
        group2_results.append(result)

    return group1_results, group2_results


def statistical_analysis(group1_results, group2_results, group1_name, group2_name):
    """Perform statistical comparison."""
    group1_means = [r['metrics']['mean_reward'] for r in group1_results]
    group2_means = [r['metrics']['mean_reward'] for r in group2_results]

    t_stat, p_value = stats.ttest_rel(group1_means, group2_means)
    cohens_d = (np.mean(group1_means) - np.mean(group2_means)) / np.sqrt(
        (np.std(group1_means)**2 + np.std(group2_means)**2) / 2
    )

    effect_size = 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large')
    significant = p_value < 0.05

    if significant:
        winner = group1_name if np.mean(group1_means) > np.mean(group2_means) else group2_name
    else:
        winner = "No significant difference"

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'effect_size': effect_size,
        'significant': bool(significant),
        'winner': winner,
        'group1_mean': float(np.mean(group1_means)),
        'group2_mean': float(np.mean(group2_means)),
        'group1_std': float(np.std(group1_means)),
        'group2_std': float(np.std(group2_means))
    }


def main():
    """Run all H2 experiments."""
    print("=" * 70)
    print("H2 HYPOTHESIS TESTING")
    print("=" * 70)

    results_summary = {}
    config_base = Path(__file__).resolve().parent.parent / "scenarios" / "configs"

    # ===================================================================
    # Experiment 1: H2-MaxPressure vs H1-Basic (Low Variance)
    # ===================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: H2-MaxPressure vs H1-Basic (Low Variance)")
    print("Hypothesis: H2 should learn faster with better reward signal")
    print("=" * 70)

    h2_results, h1_results = run_comparison(
        'h2_maxpressure',
        'h1_basic',
        str(config_base / "config_low_variance.json"),
        scenario_name='low_variance',
        num_runs=10,
        episodes=50
    )

    stats1 = statistical_analysis(h2_results, h1_results, "H2-MaxPressure", "H1-Basic")
    results_summary['experiment_1'] = {
        'description': "H2-MaxPressure vs H1-Basic on low-variance",
        'statistics': stats1,
        'h2_maxpressure_means': [r['metrics']['mean_reward'] for r in h2_results],
        'h1_basic_means': [r['metrics']['mean_reward'] for r in h1_results]
    }

    # ===================================================================
    # Experiment 2: H2-MaxPressure vs MaxPressure-Baseline (Low Variance)
    # ===================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: H2-MaxPressure vs MaxPressure-Baseline (Low Variance)")
    print("Hypothesis: H2 should approach MaxPressure heuristic performance")
    print("=" * 70)

    h2_results2, mp_baseline_results = run_comparison(
        'h2_maxpressure',
        'maxpressure_baseline',
        str(config_base / "config_low_variance.json"),
        scenario_name='low_variance',
        num_runs=10,
        episodes=50
    )

    stats2 = statistical_analysis(h2_results2, mp_baseline_results, "H2-MaxPressure", "MaxPressure-Baseline")
    results_summary['experiment_2'] = {
        'description': "H2-MaxPressure vs MaxPressure-Baseline on low-variance",
        'statistics': stats2,
        'h2_maxpressure_means': [r['metrics']['mean_reward'] for r in h2_results2],
        'maxpressure_baseline_means': [r['metrics']['mean_reward'] for r in mp_baseline_results]
    }

    # ===================================================================
    # Experiment 3: H2-MaxPressure vs H1-Basic (High Variance)
    # ===================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: H2-MaxPressure vs H1-Basic (High Variance)")
    print("Hypothesis: H2 should handle high-variance better")
    print("=" * 70)

    h2_results3, h1_results3 = run_comparison(
        'h2_maxpressure',
        'h1_basic',
        str(config_base / "config_high_variance.json"),
        scenario_name='high_variance',
        num_runs=10,
        episodes=50
    )

    stats3 = statistical_analysis(h2_results3, h1_results3, "H2-MaxPressure", "H1-Basic")
    results_summary['experiment_3'] = {
        'description': "H2-MaxPressure vs H1-Basic on high-variance",
        'statistics': stats3,
        'h2_maxpressure_means': [r['metrics']['mean_reward'] for r in h2_results3],
        'h1_basic_means': [r['metrics']['mean_reward'] for r in h1_results3]
    }

    # ===================================================================
    # Print Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    for exp_name, exp_data in results_summary.items():
        stats = exp_data['statistics']
        print(f"\n{exp_name}: {exp_data['description']}")
        
        group_names = exp_data['description'].split(' vs ')
        g1_name = group_names[0].split(' on ')[0]
        g2_name = group_names[1].split(' on ')[0] if len(group_names) > 1 else "Group2"
        
        print(f"  {g1_name}: {stats['group1_mean']:.2f} ± {stats['group1_std']:.2f}")
        print(f"  {g2_name}: {stats['group2_mean']:.2f} ± {stats['group2_std']:.2f}")
        print(f"  t={stats['t_statistic']:.3f}, p={stats['p_value']:.4f}, d={stats['cohens_d']:.3f} ({stats['effect_size']})")
        print(f"  Result: {stats['winner']} {'✅' if stats['significant'] else '⚠️'}")

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "h2_results_fixed.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✅ Results saved to {results_dir / 'h2_results_fixed.json'}")

    # Create plots
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    experiments = [
        ('experiment_1', ['H2-MP', 'H1-Basic'], ['h2_maxpressure_means', 'h1_basic_means']),
        ('experiment_2', ['H2-MP', 'MP-Base'], ['h2_maxpressure_means', 'maxpressure_baseline_means']),
        ('experiment_3', ['H2-MP', 'H1-Basic'], ['h2_maxpressure_means', 'h1_basic_means'])
    ]

    for idx, (exp_name, labels, keys) in enumerate(experiments):
        exp_data = results_summary[exp_name]
        data = [exp_data[key] for key in keys]
        
        bp = plt.boxplot(data, labels=labels, patch_artist=True, ax=axes[idx])
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        stats = exp_data['statistics']
        axes[idx].set_title(f"{exp_data['description']}\np={stats['p_value']:.4f}")
        axes[idx].set_ylabel("Mean Reward")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "h2_comparison_fixed.png", dpi=150)
    print(f"✅ Plots saved to {plots_dir / 'h2_comparison_fixed.png'}")

    print("\n" + "=" * 70)
    print("H2 EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
