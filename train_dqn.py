import torch
import matplotlib.pyplot as plt
import numpy as np
from models import CityFlowEnv, SingleAgentBaseline, EnhancedSingleAgent
from models2 import FixedLightBaseline

def smooth_curve(values, window=10):
    """Smooth reward curve for better visualization."""
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode='valid')

def train_and_evaluate():
    print("=" * 60)
    print("TRAINING DQN AGENT")
    print("=" * 60)

    # Initialize environment with aligned configuration
    # frame_skip=1 and max_steps=1000 to match baseline
    env = CityFlowEnv("data/config.json", frame_skip=1, max_steps=1000)

    # Create DQN agent
    print("\n[1/4] Creating DQN agent...")
    agent = SingleAgentBaseline(env)

    # Train the agent
    print("\n[2/4] Training DQN agent (30 episodes)...")
    print("This may take several minutes...\n")
    training_rewards = agent.train(
        episodes=30,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        decay=0.995
    )

    # Save the trained model
    print("\n[3/4] Saving trained model...")
    torch.save(agent.model.state_dict(), "dqn_model.pth")
    print("Model saved to dqn_model.pth")

    # Evaluate the trained agent
    print("\n[4/4] Evaluating trained DQN agent...")
    eval_rewards = agent.evaluate(episodes=5)
    avg_total_reward = np.mean(eval_rewards)
    # Convert to average per-step reward for fair comparison
    avg_per_step_reward = avg_total_reward / env.max_steps
    print(f"Average total episode reward: {avg_total_reward:.2f}")
    print(f"Average per-step reward: {avg_per_step_reward:.2f}")

    # Compare with baseline
    print("\n" + "=" * 60)
    print("RUNNING BASELINE FOR COMPARISON")
    print("=" * 60)
    baseline = FixedLightBaseline()
    baseline_rewards, baseline_metrics = baseline.run()
    baseline_avg_per_step = np.mean(baseline_rewards)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nDQN Average Per-Step Reward: {avg_per_step_reward:.2f}")
    print(f"Baseline Average Per-Step Reward: {baseline_avg_per_step:.2f}")
    print(f"\nBaseline Metrics:")
    for key, val in baseline_metrics.items():
        print(f"  {key}: {val:.2f}")

    # Plot training progress
    plt.figure(figsize=(15, 5))

    # Plot 1: Training rewards
    plt.subplot(1, 3, 1)
    plt.plot(training_rewards, alpha=0.3, label='Raw')
    if len(training_rewards) > 10:
        plt.plot(smooth_curve(training_rewards), label='Smoothed')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Progress")
    plt.legend()
    plt.grid(True)

    # Plot 2: Comparison of reward curves
    plt.subplot(1, 3, 2)
    plt.plot(baseline_rewards, label='FixedLightBaseline', alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Baseline Performance Over Time")
    plt.legend()
    plt.grid(True)

    # Plot 3: Average reward comparison
    plt.subplot(1, 3, 3)
    models = ['DQN (Trained)', 'Fixed Baseline']
    avg_rewards = [avg_per_step_reward, baseline_avg_per_step]
    colors = ['green' if avg_rewards[0] > avg_rewards[1] else 'orange', 'blue']
    plt.bar(models, avg_rewards, color=colors)
    plt.ylabel("Average Per-Step Reward")
    plt.title("Model Comparison")
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig("dqn_training_results.png")
    print("\nPlot saved to dqn_training_results.png")
    plt.show()

    # Determine if DQN beat the baseline
    print("\n" + "=" * 60)
    if avg_per_step_reward > baseline_avg_per_step:
        improvement = ((avg_per_step_reward - baseline_avg_per_step) / abs(baseline_avg_per_step)) * 100
        print(f"✅ SUCCESS! DQN beat the baseline by {improvement:.1f}%")
    else:
        gap = ((baseline_avg_per_step - avg_per_step_reward) / abs(baseline_avg_per_step)) * 100
        print(f"⚠️  DQN did not beat the baseline yet.")
        print(f"   Baseline is {gap:.1f}% better than DQN")
        print("   Consider: more training episodes, hyperparameter tuning, or enhanced architecture")
    print("=" * 60)

if __name__ == "__main__":
    train_and_evaluate()
