"""
Quick comparison: H1 vs H3 on 2-intersection corridor.

Demonstrates H3's scalability advantage with multiple intersections.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import torch
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.h1_basic import CityFlowEnv, H1BasicAgent
from models.h3_multiagent import MultiAgentEnv, H3MultiAgentSystem


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


def run_single_experiment(model_type, config_path, seed, episodes=50):
    """Run a single experiment and return results."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"  Run with seed={seed}...", end="")

    if model_type == 'h1_basic':
        env = CityFlowEnv(config_path, frame_skip=1, max_steps=1000)
        agent = H1BasicAgent(env, learning_rate=1e-3)

        start_time = time.time()
        training_rewards = agent.train(episodes=episodes, epsilon_start=0.951, epsilon_end=0.1, decay=0.995)
        training_time = time.time() - start_time

        # Evaluation
        env_fresh = CityFlowEnv(config_path, frame_skip=1, max_steps=1000)
        agent.model.eval()
        state = env_fresh.reset()
        eval_reward = 0
        for _ in range(1000):
            with torch.no_grad():
                q_vals = agent.model(torch.tensor(state).float().unsqueeze(0))
                action = q_vals.argmax().item()
            state, reward, done, _ = env_fresh.step(action)
            eval_reward += reward
            if done:
                break
        agent.model.train()

        param_count = count_parameters(agent)
        print(f" Eval={eval_reward:.2f}, Time={training_time:.1f}s, Params={param_count}")

    elif model_type == 'h3_multiagent':
        env = MultiAgentEnv(config_path, frame_skip=1, max_steps=1000)
        agent = H3MultiAgentSystem(env, learning_rate=1e-3)

        start_time = time.time()
        training_rewards = agent.train(episodes=episodes, epsilon_start=0.951, epsilon_end=0.1, decay=0.995)
        training_time = time.time() - start_time

        # Evaluation
        env_fresh = MultiAgentEnv(config_path, frame_skip=1, max_steps=1000)
        for agent_obj in agent.agents.values():
            agent_obj.model.eval()

        states = env_fresh.reset()
        eval_reward = 0
        for _ in range(1000):
            with torch.no_grad():
                actions = {}
                for iid, agent_obj in agent.agents.items():
                    q_vals = agent_obj.model(torch.tensor(states[iid]).float().unsqueeze(0))
                    actions[iid] = q_vals.argmax().item()

            states, rewards, done, _ = env_fresh.step(actions)
            eval_reward += sum(rewards.values())
            if done:
                break

        for agent_obj in agent.agents.values():
            agent_obj.model.train()

        param_count = count_parameters(agent)
        num_agents = len(agent.agents)
        print(f" Eval={eval_reward:.2f}, Time={training_time:.1f}s, Params={param_count} ({num_agents} agents)")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return {
        'eval_reward': eval_reward,
        'training_time': training_time,
        'param_count': param_count,
        'training_rewards': training_rewards
    }


def main():
    """Run quick corridor comparison."""
    print("=" * 70)
    print("H1 vs H3 COMPARISON ON 2-INTERSECTION CORRIDOR")
    print("=" * 70)

    config_path = "scenarios/configs/config_2intersection_corridor.json"
    num_runs = 5
    episodes = 50

    # H1-Basic runs
    print(f"\nRunning H1-Basic ({num_runs} runs, {episodes} episodes each)...")
    h1_results = []
    for i in range(num_runs):
        result = run_single_experiment('h1_basic', config_path, seed=42+i, episodes=episodes)
        h1_results.append(result)

    # H3-MultiAgent runs
    print(f"\nRunning H3-MultiAgent ({num_runs} runs, {episodes} episodes each)...")
    h3_results = []
    for i in range(num_runs):
        result = run_single_experiment('h3_multiagent', config_path, seed=42+i, episodes=episodes)
        h3_results.append(result)

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    h1_rewards = [r['eval_reward'] for r in h1_results]
    h3_rewards = [r['eval_reward'] for r in h3_results]

    h1_time = np.mean([r['training_time'] for r in h1_results])
    h3_time = np.mean([r['training_time'] for r in h3_results])

    h1_params = h1_results[0]['param_count']
    h3_params = h3_results[0]['param_count']

    print(f"\nH1-Basic (Single Agent for 2 intersections):")
    print(f"  Evaluation Reward: {np.mean(h1_rewards):.2f} ± {np.std(h1_rewards):.2f}")
    print(f"  Training Time: {h1_time:.1f}s")
    print(f"  Parameters: {h1_params:,}")

    print(f"\nH3-MultiAgent (2 Independent Agents):")
    print(f"  Evaluation Reward: {np.mean(h3_rewards):.2f} ± {np.std(h3_rewards):.2f}")
    print(f"  Training Time: {h3_time:.1f}s")
    print(f"  Parameters: {h3_params:,}")

    print(f"\nScalability Comparison:")
    print(f"  Parameter Reduction: {(1 - h3_params/h1_params)*100:.1f}% ({h1_params/h3_params:.2f}x fewer in H3)")
    print(f"  Training Speedup: {(1 - h3_time/h1_time)*100:.1f}% ({h1_time/h3_time:.2f}x faster in H3)")

    # Statistical test
    t_stat, p_value = stats.ttest_rel(h1_rewards, h3_rewards)
    print(f"\nStatistical Test (Paired t-test):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        winner = "H1-Basic" if np.mean(h1_rewards) > np.mean(h3_rewards) else "H3-MultiAgent"
        print(f"  Result: {winner} is significantly better (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    summary = {
        'h1_basic': {
            'mean_reward': float(np.mean(h1_rewards)),
            'std_reward': float(np.std(h1_rewards)),
            'mean_time': float(h1_time),
            'params': int(h1_params)
        },
        'h3_multiagent': {
            'mean_reward': float(np.mean(h3_rewards)),
            'std_reward': float(np.std(h3_rewards)),
            'mean_time': float(h3_time),
            'params': int(h3_params),
            'num_agents': 2
        },
        'comparison': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'param_reduction_pct': float((1 - h3_params/h1_params)*100),
            'time_speedup_pct': float((1 - h3_time/h1_time)*100)
        }
    }

    with open(results_dir / "corridor_comparison.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved to {results_dir / 'corridor_comparison.json'}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Performance comparison
    axes[0].bar(['H1-Basic', 'H3-MultiAgent'], [np.mean(h1_rewards), np.mean(h3_rewards)],
                yerr=[np.std(h1_rewards), np.std(h3_rewards)], capsize=5,
                color=['lightblue', 'lightgreen'])
    axes[0].set_ylabel('Evaluation Reward')
    axes[0].set_title('Performance Comparison')
    axes[0].grid(axis='y', alpha=0.3)

    # Parameter efficiency
    axes[1].bar(['H1-Basic', 'H3-MultiAgent'], [h1_params, h3_params],
                color=['lightblue', 'lightgreen'])
    axes[1].set_ylabel('Number of Parameters')
    axes[1].set_title('Model Complexity')
    axes[1].grid(axis='y', alpha=0.3)

    # Training time
    axes[2].bar(['H1-Basic', 'H3-MultiAgent'], [h1_time, h3_time],
                color=['lightblue', 'lightgreen'])
    axes[2].set_ylabel('Training Time (s)')
    axes[2].set_title('Training Efficiency')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'plots' / 'corridor_comparison.png', dpi=150)
    print(f"✅ Plot saved to {results_dir / 'plots' / 'corridor_comparison.png'}")

    print("\n" + "=" * 70)
    print("CORRIDOR COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
