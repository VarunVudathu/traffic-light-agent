"""
H3 Experiment Runner with Statistical Analysis.

Tests H3 hypothesis:
"Multi-agent coordination using independent DQN agents per intersection will improve
scalability and reduce computational overhead compared to single-agent approaches."

Experiments:
1. H3-MultiAgent vs H1-Basic on low-variance traffic
2. H3-MultiAgent vs H1-Basic on high-variance traffic
3. Scalability analysis (parameter count, training time)
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
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.h1_basic import CityFlowEnv, H1BasicAgent
from models.h3_multiagent import MultiAgentEnv, H3MultiAgentSystem
from utils.logger import EnhancedLogger


def compute_metrics(rewards, max_steps):
    """Compute performance metrics from reward history."""
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'total_reward': np.sum(rewards),
        'final_100_mean': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    }


def evaluate_with_logging(agent, env, scenario_name, max_steps=1000, is_multiagent=False):
    """
    Evaluate agent with enhanced logging.

    Args:
        agent: Trained agent (H1BasicAgent or H3MultiAgentSystem)
        env: Environment (CityFlowEnv or MultiAgentEnv)
        scenario_name: Name of scenario
        max_steps: Max simulation steps
        is_multiagent: Whether this is H3 multi-agent system

    Returns:
        Dict with enhanced metrics
    """
    logger = EnhancedLogger()
    rewards = []

    if is_multiagent:
        # H3 Multi-agent evaluation
        for agent_obj in agent.agents.values():
            agent_obj.model.eval()

        states = env.reset()

        for step in range(max_steps):
            with torch.no_grad():
                actions = {}
                for iid, agent_obj in agent.agents.items():
                    q_vals = agent_obj.model(torch.tensor(states[iid]).float().unsqueeze(0))
                    actions[iid] = q_vals.argmax().item()

            next_states, agent_rewards, done, _ = env.step(actions)
            total_reward = sum(agent_rewards.values())
            rewards.append(total_reward)
            logger.update(env.eng, current_step=step)

            states = next_states
            if done:
                break

        for agent_obj in agent.agents.values():
            agent_obj.model.train()

    else:
        # H1-Basic single-agent evaluation
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

    enhanced_metrics = logger.get_enhanced_metrics(scenario_name=scenario_name)
    enhanced_metrics['rewards'] = rewards
    enhanced_metrics['mean_reward'] = np.mean(rewards)
    return enhanced_metrics


def count_parameters(model_or_system):
    """Count trainable parameters in model(s)."""
    if hasattr(model_or_system, 'agents'):
        # H3 multi-agent system
        total = 0
        for agent in model_or_system.agents.values():
            total += sum(p.numel() for p in agent.model.parameters() if p.requires_grad)
        return total
    else:
        # Single agent
        return sum(p.numel() for p in model_or_system.model.parameters() if p.requires_grad)


def run_single_experiment(model_type, config_path, seed, scenario_name='', episodes=50):
    """
    Run a single experiment.

    Args:
        model_type: 'h1_basic' or 'h3_multiagent'
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

        start_time = time.time()
        training_rewards = agent.train(
            episodes=episodes,
            epsilon_start=0.951,
            epsilon_end=0.1,
            decay=0.995
        )
        training_time = time.time() - start_time

        env_fresh = CityFlowEnv(config_path, frame_skip=1, max_steps=1000)
        enhanced_metrics = evaluate_with_logging(agent, env_fresh, scenario_name, max_steps=1000, is_multiagent=False)
        param_count = count_parameters(agent)

    elif model_type == 'h3_multiagent':
        env = MultiAgentEnv(config_path, frame_skip=1, max_steps=1000)
        agent = H3MultiAgentSystem(env, learning_rate=1e-3)

        start_time = time.time()
        training_rewards = agent.train(
            episodes=episodes,
            epsilon_start=0.951,
            epsilon_end=0.1,
            decay=0.995
        )
        training_time = time.time() - start_time

        env_fresh = MultiAgentEnv(config_path, frame_skip=1, max_steps=1000)
        enhanced_metrics = evaluate_with_logging(agent, env_fresh, scenario_name, max_steps=1000, is_multiagent=True)
        param_count = count_parameters(agent)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    eval_rewards = enhanced_metrics['rewards']
    metrics = compute_metrics(eval_rewards, 1000)
    metrics['training_time'] = training_time
    metrics['param_count'] = param_count

    print(f"  Mean reward: {metrics['mean_reward']:.2f}, Time: {training_time:.1f}s, Params: {param_count}")

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

    # Compute efficiency metrics
    group1_time = np.mean([r['metrics']['training_time'] for r in group1_results])
    group2_time = np.mean([r['metrics']['training_time'] for r in group2_results])
    group1_params = group1_results[0]['metrics']['param_count']
    group2_params = group2_results[0]['metrics']['param_count']

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
        'group2_std': float(np.std(group2_means)),
        'group1_training_time': float(group1_time),
        'group2_training_time': float(group2_time),
        'group1_param_count': int(group1_params),
        'group2_param_count': int(group2_params)
    }


def main():
    """Run all H3 experiments."""
    print("=" * 70)
    print("H3 HYPOTHESIS TESTING - MULTI-AGENT COORDINATION")
    print("=" * 70)

    results_summary = {}
    config_base = Path(__file__).resolve().parent.parent / "scenarios" / "configs"

    # ===================================================================
    # Experiment 1: H3-MultiAgent vs H1-Basic (Low Variance)
    # ===================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: H3-MultiAgent vs H1-Basic (Low Variance)")
    print("Hypothesis: Multi-agent should scale better with independent learning")
    print("=" * 70)

    h3_results, h1_results = run_comparison(
        'h3_multiagent',
        'h1_basic',
        str(config_base / "config_low_variance.json"),
        scenario_name='low_variance',
        num_runs=10,
        episodes=50
    )

    stats1 = statistical_analysis(h3_results, h1_results, "H3-MultiAgent", "H1-Basic")
    results_summary['experiment_1'] = {
        'description': "H3-MultiAgent vs H1-Basic on low-variance",
        'statistics': stats1,
        'h3_multiagent_means': [r['metrics']['mean_reward'] for r in h3_results],
        'h1_basic_means': [r['metrics']['mean_reward'] for r in h1_results]
    }

    # ===================================================================
    # Experiment 2: H3-MultiAgent vs H1-Basic (High Variance)
    # ===================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: H3-MultiAgent vs H1-Basic (High Variance)")
    print("Hypothesis: Multi-agent should handle complex scenarios better")
    print("=" * 70)

    h3_results2, h1_results2 = run_comparison(
        'h3_multiagent',
        'h1_basic',
        str(config_base / "config_high_variance.json"),
        scenario_name='high_variance',
        num_runs=10,
        episodes=50
    )

    stats2 = statistical_analysis(h3_results2, h1_results2, "H3-MultiAgent", "H1-Basic")
    results_summary['experiment_2'] = {
        'description': "H3-MultiAgent vs H1-Basic on high-variance",
        'statistics': stats2,
        'h3_multiagent_means': [r['metrics']['mean_reward'] for r in h3_results2],
        'h1_basic_means': [r['metrics']['mean_reward'] for r in h1_results2]
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

        # Print efficiency metrics
        print(f"\n  Efficiency Comparison:")
        print(f"    {g1_name}: {stats['group1_param_count']:,} params, {stats['group1_training_time']:.1f}s training")
        print(f"    {g2_name}: {stats['group2_param_count']:,} params, {stats['group2_training_time']:.1f}s training")
        param_ratio = stats['group1_param_count'] / max(stats['group2_param_count'], 1)
        time_ratio = stats['group1_training_time'] / max(stats['group2_training_time'], 1)
        print(f"    Param ratio: {param_ratio:.2f}x, Time ratio: {time_ratio:.2f}x")

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "h3_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✅ Results saved to {results_dir / 'h3_results.json'}")

    # Create plots
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    experiments = [
        ('experiment_1', ['H3-Multi', 'H1-Basic'], ['h3_multiagent_means', 'h1_basic_means']),
        ('experiment_2', ['H3-Multi', 'H1-Basic'], ['h3_multiagent_means', 'h1_basic_means'])
    ]

    for idx, (exp_name, labels, keys) in enumerate(experiments):
        exp_data = results_summary[exp_name]
        data = [exp_data[key] for key in keys]

        bp = axes[idx].boxplot(data, labels=labels, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen' if idx == 0 else 'lightcoral')

        stats = exp_data['statistics']
        axes[idx].set_title(f"{exp_data['description']}\\np={stats['p_value']:.4f}")
        axes[idx].set_ylabel("Mean Reward")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "h3_comparison.png", dpi=150)
    print(f"✅ Plots saved to {plots_dir / 'h3_comparison.png'}")

    print("\n" + "=" * 70)
    print("H3 EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
