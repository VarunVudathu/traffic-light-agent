"""
H1 Experiment Runner with Statistical Analysis.

Tests H1 hypothesis:
"A Single-Agent DQN using standard queue metrics will outperform Fixed-Time
controllers in low-variance traffic, but will fail to adapt to high-variance
surges due to lack of future-aware state features."

Experiments:
1. H1-Basic vs Fixed-Time on low-variance traffic (should win)
2. H1-Basic vs H1-Enhanced on high-variance/surge traffic (should lose)
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
from models.h1_enhanced import CityFlowEnvEnhanced, H1EnhancedAgent
from models.baselines import FixedTimeBaseline
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


def evaluate_with_logging(agent, env, scenario_name, max_steps=1000):
    """
    Evaluate agent with enhanced logging to collect detailed metrics.

    Args:
        agent: Trained agent (or None for baseline)
        env: Environment or baseline controller
        scenario_name: Name of scenario for surge-specific metrics
        max_steps: Max simulation steps

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
            # Select action
            with torch.no_grad():
                q_vals = agent.model(torch.tensor(state).float().unsqueeze(0))
                action = q_vals.argmax().item()

            # Take step
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            # Update logger
            logger.update(env.eng, current_step=step)

            state = next_state
            if done:
                break

        agent.model.train()

    # For baseline (FixedTimeBaseline)
    else:
        # Run baseline and track with logger
        # Need to access engine during simulation
        # Since baseline.run() doesn't expose step-by-step access,
        # we'll manually run it with logging
        baseline = env  # env is actually the baseline object
        baseline.eng.reset()

        for step in range(max_steps):
            # Update traffic lights (baseline logic)
            for iid, state in baseline.intersection_state.items():
                phases = baseline.eng.get_intersection_phase(iid)
                if not phases:
                    continue

                state['time_in_phase'] += 1
                if state['time_in_phase'] >= baseline.phase_duration:
                    state['current_phase'] = (state['current_phase'] + 1) % len(phases)
                    state['time_in_phase'] = 0

                baseline.eng.set_tl_phase(iid, state['current_phase'])

            # Step simulation
            baseline.eng.next_step()

            # Compute reward
            waiting_counts = list(baseline.eng.get_lane_waiting_vehicle_count().values())
            reward = -np.mean(waiting_counts) if waiting_counts else 0.0
            rewards.append(reward)

            # Update logger
            logger.update(baseline.eng, current_step=step)

    # Get enhanced metrics
    enhanced_metrics = logger.get_enhanced_metrics(scenario_name=scenario_name)
    enhanced_metrics['rewards'] = rewards
    enhanced_metrics['mean_reward'] = np.mean(rewards)

    return enhanced_metrics


def paired_ttest(group1, group2, name1="Group1", name2="Group2"):
    """
    Perform paired t-test and compute effect size.

    Returns: dict with t-statistic, p-value, Cohen's d, interpretation
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(group1, group2)

    # Cohen's d for paired samples
    diff = np.array(group1) - np.array(group2)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"

    # Determine winner
    if p_value < 0.05:
        winner = name1 if np.mean(group1) > np.mean(group2) else name2
        significant = True
    else:
        winner = "No significant difference"
        significant = False

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_size': effect,
        'significant': significant,
        'winner': winner,
        'group1_mean': np.mean(group1),
        'group2_mean': np.mean(group2),
        'group1_std': np.std(group1),
        'group2_std': np.std(group2)
    }


def run_single_experiment(model_type, config_path, seed, scenario_name='', episodes=50):
    """
    Run a single experiment with given configuration.

    Args:
        model_type: 'h1_basic', 'h1_enhanced', or 'fixed_time'
        config_path: Path to CityFlow config
        seed: Random seed
        scenario_name: Name of scenario for metrics (e.g., 'extreme_surge')
        episodes: Number of training episodes (for DQN models)

    Returns:
        dict with rewards and metrics (including enhanced metrics)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model_type == 'fixed_time':
        baseline = FixedTimeBaseline(config_path, phase_duration=30, max_steps=1000)
        rewards = baseline.run()
        metrics = compute_metrics(rewards, 1000)

        # Collect enhanced metrics
        baseline_fresh = FixedTimeBaseline(config_path, phase_duration=30, max_steps=1000)
        enhanced_metrics = evaluate_with_logging(None, baseline_fresh, scenario_name, max_steps=1000)

        return {
            'training_rewards': None,
            'eval_rewards': rewards,
            'metrics': metrics,
            'enhanced_metrics': enhanced_metrics
        }

    elif model_type == 'h1_basic':
        env = CityFlowEnv(config_path, frame_skip=1, max_steps=1000)
        agent = H1BasicAgent(env)
        training_rewards = agent.train(episodes=episodes, epsilon_start=1.0, epsilon_end=0.1)

        # Evaluate
        eval_rewards_list = []
        for _ in range(5):  # 5 evaluation episodes
            eval_rewards_list.extend(agent.evaluate(episodes=1))

        eval_rewards = eval_rewards_list
        metrics = compute_metrics(eval_rewards, 1000)

        # Collect enhanced metrics
        env_fresh = CityFlowEnv(config_path, frame_skip=1, max_steps=1000)
        enhanced_metrics = evaluate_with_logging(agent, env_fresh, scenario_name, max_steps=1000)

        return {
            'training_rewards': training_rewards,
            'eval_rewards': eval_rewards,
            'metrics': metrics,
            'enhanced_metrics': enhanced_metrics
        }

    elif model_type == 'h1_enhanced':
        env = CityFlowEnvEnhanced(config_path, frame_skip=1, max_steps=1000, history_length=4)
        agent = H1EnhancedAgent(env)
        training_rewards = agent.train(episodes=episodes, epsilon_start=1.0, epsilon_end=0.1)

        # Evaluate
        eval_rewards_list = []
        for _ in range(5):
            eval_rewards_list.extend(agent.evaluate(episodes=1))

        eval_rewards = eval_rewards_list
        metrics = compute_metrics(eval_rewards, 1000)

        # Collect enhanced metrics
        env_fresh = CityFlowEnvEnhanced(config_path, frame_skip=1, max_steps=1000, history_length=4)
        enhanced_metrics = evaluate_with_logging(agent, env_fresh, scenario_name, max_steps=1000)

        return {
            'training_rewards': training_rewards,
            'eval_rewards': eval_rewards,
            'metrics': metrics,
            'enhanced_metrics': enhanced_metrics
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_experiment_suite(model_type, scenario_name, config_path, num_runs=10, episodes=50):
    """
    Run multiple experiments with different seeds.

    Returns: list of results from each run
    """
    print(f"\n{'='*70}")
    print(f"Running: {model_type} on {scenario_name}")
    print(f"{'='*70}")

    results = []
    for run in range(num_runs):
        seed = 42 + run
        print(f"Run {run+1}/{num_runs} (seed={seed})...")

        result = run_single_experiment(model_type, config_path, seed, scenario_name, episodes)
        results.append(result)

        # Print quick summary
        if result['metrics']:
            print(f"  Mean reward: {result['metrics']['mean_reward']:.2f}")
            if 'enhanced_metrics' in result and 'recovery_rate_mean' in result['enhanced_metrics']:
                print(f"  Recovery rate: {result['enhanced_metrics']['recovery_rate_mean']:.1f} steps")
            if 'enhanced_metrics' in result and 'peak_queue_mean' in result['enhanced_metrics']:
                print(f"  Peak queue: {result['enhanced_metrics']['peak_queue_mean']:.1f}")

    return results


def main():
    """Run H1 experiments and statistical analysis."""
    print("\n" + "="*70)
    print("H1 HYPOTHESIS TESTING")
    print("="*70)

    # Configuration
    num_runs = 10
    training_episodes = 50

    # Scenarios
    scenarios_dir = Path(__file__).resolve().parent.parent / 'scenarios' / 'configs'
    scenarios = {
        'low_variance': str(scenarios_dir / 'config_low_variance.json'),
        'high_variance': str(scenarios_dir / 'config_high_variance.json'),
        'moderate_surge': str(scenarios_dir / 'config_moderate_surge.json'),
        'extreme_surge': str(scenarios_dir / 'config_extreme_surge.json')
    }

    all_results = {}

    # Experiment 1: H1-Basic vs Fixed-Time on low-variance
    print("\n" + "="*70)
    print("EXPERIMENT 1: H1-Basic vs Fixed-Time (Low Variance)")
    print("Hypothesis: H1-Basic should outperform Fixed-Time")
    print("="*70)

    all_results['exp1_h1_basic'] = run_experiment_suite(
        'h1_basic', 'low_variance', scenarios['low_variance'], num_runs, training_episodes
    )
    all_results['exp1_fixed_time'] = run_experiment_suite(
        'fixed_time', 'low_variance', scenarios['low_variance'], num_runs, 0
    )

    # Experiment 2a: H1-Basic vs H1-Enhanced on high-variance
    print("\n" + "="*70)
    print("EXPERIMENT 2a: H1-Basic vs H1-Enhanced (High Variance)")
    print("Hypothesis: H1-Enhanced should outperform H1-Basic")
    print("="*70)

    all_results['exp2a_h1_basic'] = run_experiment_suite(
        'h1_basic', 'high_variance', scenarios['high_variance'], num_runs, training_episodes
    )
    all_results['exp2a_h1_enhanced'] = run_experiment_suite(
        'h1_enhanced', 'high_variance', scenarios['high_variance'], num_runs, training_episodes
    )

    # Experiment 2b: H1-Basic vs H1-Enhanced on surge
    print("\n" + "="*70)
    print("EXPERIMENT 2b: H1-Basic vs H1-Enhanced (Extreme Surge)")
    print("Hypothesis: H1-Enhanced should handle surges better")
    print("="*70)

    all_results['exp2b_h1_basic'] = run_experiment_suite(
        'h1_basic', 'extreme_surge', scenarios['extreme_surge'], num_runs, training_episodes
    )
    all_results['exp2b_h1_enhanced'] = run_experiment_suite(
        'h1_enhanced', 'extreme_surge', scenarios['extreme_surge'], num_runs, training_episodes
    )

    # Statistical Analysis
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    # Extract mean rewards for statistical tests
    exp1_h1_basic_means = [r['metrics']['mean_reward'] for r in all_results['exp1_h1_basic']]
    exp1_fixed_means = [r['metrics']['mean_reward'] for r in all_results['exp1_fixed_time']]

    exp2a_h1_basic_means = [r['metrics']['mean_reward'] for r in all_results['exp2a_h1_basic']]
    exp2a_h1_enhanced_means = [r['metrics']['mean_reward'] for r in all_results['exp2a_h1_enhanced']]

    exp2b_h1_basic_means = [r['metrics']['mean_reward'] for r in all_results['exp2b_h1_basic']]
    exp2b_h1_enhanced_means = [r['metrics']['mean_reward'] for r in all_results['exp2b_h1_enhanced']]

    # Perform statistical tests
    stats_exp1 = paired_ttest(exp1_h1_basic_means, exp1_fixed_means, "H1-Basic", "Fixed-Time")
    stats_exp2a = paired_ttest(exp2a_h1_enhanced_means, exp2a_h1_basic_means, "H1-Enhanced", "H1-Basic")
    stats_exp2b = paired_ttest(exp2b_h1_enhanced_means, exp2b_h1_basic_means, "H1-Enhanced", "H1-Basic")

    # Print results
    print("\nExperiment 1: H1-Basic vs Fixed-Time (Low Variance)")
    print(f"  H1-Basic:   {stats_exp1['group1_mean']:.2f} ± {stats_exp1['group1_std']:.2f}")
    print(f"  Fixed-Time: {stats_exp1['group2_mean']:.2f} ± {stats_exp1['group2_std']:.2f}")
    print(f"  t={stats_exp1['t_statistic']:.3f}, p={stats_exp1['p_value']:.4f}, d={stats_exp1['cohens_d']:.3f} ({stats_exp1['effect_size']})")
    print(f"  Result: {stats_exp1['winner']} {'✅' if stats_exp1['significant'] else '⚠️'}")

    print("\nExperiment 2a: H1-Enhanced vs H1-Basic (High Variance)")
    print(f"  H1-Enhanced: {stats_exp2a['group1_mean']:.2f} ± {stats_exp2a['group1_std']:.2f}")
    print(f"  H1-Basic:    {stats_exp2a['group2_mean']:.2f} ± {stats_exp2a['group2_std']:.2f}")
    print(f"  t={stats_exp2a['t_statistic']:.3f}, p={stats_exp2a['p_value']:.4f}, d={stats_exp2a['cohens_d']:.3f} ({stats_exp2a['effect_size']})")
    print(f"  Result: {stats_exp2a['winner']} {'✅' if stats_exp2a['significant'] else '⚠️'}")

    print("\nExperiment 2b: H1-Enhanced vs H1-Basic (Extreme Surge)")
    print(f"  H1-Enhanced: {stats_exp2b['group1_mean']:.2f} ± {stats_exp2b['group1_std']:.2f}")
    print(f"  H1-Basic:    {stats_exp2b['group2_mean']:.2f} ± {stats_exp2b['group2_std']:.2f}")
    print(f"  t={stats_exp2b['t_statistic']:.3f}, p={stats_exp2b['p_value']:.4f}, d={stats_exp2b['cohens_d']:.3f} ({stats_exp2b['effect_size']})")
    print(f"  Result: {stats_exp2b['winner']} {'✅' if stats_exp2b['significant'] else '⚠️'}")

    # Collect enhanced metrics
    def extract_enhanced_metrics(results_list, metric_key):
        """Extract a specific enhanced metric from results list."""
        values = []
        for r in results_list:
            if 'enhanced_metrics' in r and metric_key in r['enhanced_metrics']:
                values.append(r['enhanced_metrics'][metric_key])
        return values if values else None

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    results_summary = {
        'experiment_1': {
            'description': 'H1-Basic vs Fixed-Time on low-variance',
            'statistics': stats_exp1,
            'h1_basic_means': exp1_h1_basic_means,
            'fixed_time_means': exp1_fixed_means,
            'enhanced_metrics': {
                'h1_basic_throughput': extract_enhanced_metrics(all_results['exp1_h1_basic'], 'throughput'),
                'fixed_time_throughput': extract_enhanced_metrics(all_results['exp1_fixed_time'], 'throughput'),
                'h1_basic_queue_variance': extract_enhanced_metrics(all_results['exp1_h1_basic'], 'queue_variance'),
                'fixed_time_queue_variance': extract_enhanced_metrics(all_results['exp1_fixed_time'], 'queue_variance'),
            }
        },
        'experiment_2a': {
            'description': 'H1-Enhanced vs H1-Basic on high-variance',
            'statistics': stats_exp2a,
            'h1_enhanced_means': exp2a_h1_enhanced_means,
            'h1_basic_means': exp2a_h1_basic_means,
            'enhanced_metrics': {
                'h1_enhanced_recovery_rate': extract_enhanced_metrics(all_results['exp2a_h1_enhanced'], 'recovery_rate_mean'),
                'h1_basic_recovery_rate': extract_enhanced_metrics(all_results['exp2a_h1_basic'], 'recovery_rate_mean'),
                'h1_enhanced_peak_queue': extract_enhanced_metrics(all_results['exp2a_h1_enhanced'], 'peak_queue_mean'),
                'h1_basic_peak_queue': extract_enhanced_metrics(all_results['exp2a_h1_basic'], 'peak_queue_mean'),
            }
        },
        'experiment_2b': {
            'description': 'H1-Enhanced vs H1-Basic on extreme surge',
            'statistics': stats_exp2b,
            'h1_enhanced_means': exp2b_h1_enhanced_means,
            'h1_basic_means': exp2b_h1_basic_means,
            'enhanced_metrics': {
                'h1_enhanced_recovery_rate': extract_enhanced_metrics(all_results['exp2b_h1_enhanced'], 'recovery_rate_mean'),
                'h1_basic_recovery_rate': extract_enhanced_metrics(all_results['exp2b_h1_basic'], 'recovery_rate_mean'),
                'h1_enhanced_peak_queue': extract_enhanced_metrics(all_results['exp2b_h1_enhanced'], 'peak_queue_mean'),
                'h1_basic_peak_queue': extract_enhanced_metrics(all_results['exp2b_h1_basic'], 'peak_queue_mean'),
                'h1_enhanced_queue_variance': extract_enhanced_metrics(all_results['exp2b_h1_enhanced'], 'queue_variance'),
                'h1_basic_queue_variance': extract_enhanced_metrics(all_results['exp2b_h1_basic'], 'queue_variance'),
            }
        }
    }

    with open(results_dir / 'h1_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✅ Results saved to {results_dir / 'h1_results.json'}")

    # Generate plots
    generate_plots(results_summary, results_dir)

    print("\n" + "="*70)
    print("H1 EXPERIMENTS COMPLETE")
    print("="*70)


def generate_plots(results_summary, results_dir):
    """Generate comparison plots."""
    plt.figure(figsize=(15, 5))

    experiments = [
        ('experiment_1', 'H1-Basic vs Fixed-Time\n(Low Variance)', ['H1-Basic', 'Fixed-Time']),
        ('experiment_2a', 'H1-Enhanced vs H1-Basic\n(High Variance)', ['H1-Enhanced', 'H1-Basic']),
        ('experiment_2b', 'H1-Enhanced vs H1-Basic\n(Extreme Surge)', ['H1-Enhanced', 'H1-Basic'])
    ]

    for idx, (exp_key, title, labels) in enumerate(experiments, 1):
        plt.subplot(1, 3, idx)

        exp_data = results_summary[exp_key]
        stats_data = exp_data['statistics']

        # Get data
        if exp_key == 'experiment_1':
            data = [exp_data['h1_basic_means'], exp_data['fixed_time_means']]
        else:
            data = [exp_data['h1_enhanced_means'], exp_data['h1_basic_means']]

        # Box plot
        bp = plt.boxplot(data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        plt.ylabel('Mean Reward')
        plt.title(title)
        plt.grid(axis='y', alpha=0.3)

        # Add significance marker
        if stats_data['significant']:
            y_max = max([max(d) for d in data])
            y_min = min([min(d) for d in data])
            y_range = y_max - y_min
            y_pos = y_max + 0.1 * y_range

            plt.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1)
            sig_marker = '***' if stats_data['p_value'] < 0.001 else '**' if stats_data['p_value'] < 0.01 else '*'
            plt.text(1.5, y_pos, sig_marker, ha='center', va='bottom', fontsize=14)

        # Add p-value text
        plt.text(0.5, 0.95, f"p={stats_data['p_value']:.4f}\nd={stats_data['cohens_d']:.3f}",
                 transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(results_dir / 'plots' / 'h1_comparison.png', dpi=150)
    print(f"✅ Plots saved to {results_dir / 'plots' / 'h1_comparison.png'}")
    plt.close()


if __name__ == '__main__':
    main()
