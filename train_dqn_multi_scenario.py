import torch
import matplotlib.pyplot as plt
import numpy as np
from models import CityFlowEnv, SingleAgentBaseline
from models2 import FixedLightBaseline
import random

def smooth_curve(values, window=10):
    """Smooth reward curve for better visualization."""
    if len(values) < window:
        return values
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode='valid')

class MultiScenarioTrainer:
    """Train DQN agent on multiple traffic scenarios."""

    def __init__(self, config_files, frame_skip=1, max_steps=1000):
        self.config_files = config_files
        self.frame_skip = frame_skip
        self.max_steps = max_steps

        # Create one environment to get state/action space size
        temp_env = CityFlowEnv(config_files[0], frame_skip, max_steps)
        self.state_size = len(temp_env.get_state())
        self.action_size = len(temp_env.action_space)

        # Initialize agent with shared model
        self.agent = SingleAgentBaseline(temp_env)

        # Tracking metrics per scenario
        self.scenario_rewards = {config: [] for config in config_files}

    def train(self, episodes=150, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, decay=0.995):
        """Train agent by alternating between scenarios."""
        epsilon = epsilon_start
        all_rewards = []

        print(f"Training on {len(self.config_files)} scenarios for {episodes} episodes")
        print(f"Scenarios: {self.config_files}\n")

        for ep in range(episodes):
            # Alternate between scenarios (or randomize)
            config_idx = ep % len(self.config_files)
            config_file = self.config_files[config_idx]
            scenario_name = config_file.split('/')[-1].replace('config-', '').replace('.json', '')

            # Create fresh environment for this scenario
            env = CityFlowEnv(config_file, self.frame_skip, self.max_steps)
            self.agent.env = env  # Update agent's environment

            # Run episode
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.choice(env.action_space)
                else:
                    with torch.no_grad():
                        q_vals = self.agent.model(torch.tensor(state).float().unsqueeze(0))
                        action = q_vals.argmax().item()

                next_state, reward, done, _ = env.step(action)
                self.agent.buffer.add((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                steps += 1

                # Update model if enough data
                if len(self.agent.buffer) > 128:  # Increased from 64
                    self.agent._update(gamma)

            # Track rewards
            all_rewards.append(total_reward)
            self.scenario_rewards[config_file].append(total_reward)

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * decay)

            # Print progress
            avg_last_10 = np.mean(all_rewards[-10:]) if len(all_rewards) >= 10 else np.mean(all_rewards)
            print(f"[Ep {ep+1:3d}/{episodes}] Scenario: {scenario_name:8s} | "
                  f"Reward: {total_reward:7.2f} | Steps: {steps:4d} | "
                  f"Avg(10): {avg_last_10:7.2f} | ε: {epsilon:.3f}")

        return all_rewards

    def evaluate(self, episodes=5):
        """Evaluate on all scenarios."""
        print("\n" + "=" * 70)
        print("EVALUATING TRAINED AGENT ON ALL SCENARIOS")
        print("=" * 70)

        results = {}
        self.agent.model.eval()

        for config_file in self.config_files:
            scenario_name = config_file.split('/')[-1].replace('config-', '').replace('.json', '')
            env = CityFlowEnv(config_file, self.frame_skip, self.max_steps)
            self.agent.env = env

            rewards = []
            with torch.no_grad():
                for ep in range(episodes):
                    state = env.reset()
                    total = 0
                    done = False
                    while not done:
                        q_vals = self.agent.model(torch.tensor(state).float().unsqueeze(0))
                        action = q_vals.argmax().item()
                        state, reward, done, _ = env.step(action)
                        total += reward
                    rewards.append(total)

            avg_reward = np.mean(rewards)
            avg_per_step = avg_reward / self.max_steps
            results[config_file] = {
                'total': avg_reward,
                'per_step': avg_per_step,
                'name': scenario_name
            }
            print(f"  {scenario_name:15s}: {avg_per_step:7.2f} per-step | {avg_reward:7.2f} total")

        return results

def train_and_evaluate():
    print("=" * 70)
    print("MULTI-SCENARIO DQN TRAINING")
    print("=" * 70)

    # Define scenarios (light and heavy traffic)
    scenarios = [
        "data/config-light.json",
        "data/config-heavy.json"
    ]

    # Create trainer
    print("\n[1/5] Creating multi-scenario trainer...")
    trainer = MultiScenarioTrainer(scenarios, frame_skip=1, max_steps=1000)

    # Train the agent
    print("\n[2/5] Training DQN agent (150 episodes across scenarios)...")
    print("This will take several minutes...\n")
    training_rewards = trainer.train(
        episodes=150,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        decay=0.995
    )

    # Save the trained model
    print("\n[3/5] Saving trained model...")
    torch.save(trainer.agent.model.state_dict(), "dqn_model_multi_scenario.pth")
    print("Model saved to dqn_model_multi_scenario.pth")

    # Evaluate the trained agent
    print("\n[4/5] Evaluating trained DQN agent...")
    eval_results = trainer.evaluate(episodes=5)

    # Compare with baseline on both scenarios
    print("\n[5/5] Running baseline for comparison...")
    baseline_results = {}
    for config_file in scenarios:
        scenario_name = config_file.split('/')[-1].replace('config-', '').replace('.json', '')

        # Temporarily modify config in models2 to use this scenario
        import models2
        original_config = models2.CONFIG["config_file"]
        models2.CONFIG["config_file"] = config_file

        baseline = FixedLightBaseline()
        baseline_rewards, _ = baseline.run()
        baseline_avg = np.mean(baseline_rewards)

        baseline_results[config_file] = {
            'per_step': baseline_avg,
            'name': scenario_name
        }

        # Restore original config
        models2.CONFIG["config_file"] = original_config

        print(f"  {scenario_name:15s}: {baseline_avg:7.2f} per-step")

    # Print comparison summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for config_file in scenarios:
        scenario_name = eval_results[config_file]['name']
        dqn_score = eval_results[config_file]['per_step']
        baseline_score = baseline_results[config_file]['per_step']

        print(f"\n{scenario_name.upper()} Scenario:")
        print(f"  DQN Baseline:   {dqn_score:7.2f}")
        print(f"  Fixed Baseline: {baseline_score:7.2f}")

        if dqn_score > baseline_score:
            improvement = ((dqn_score - baseline_score) / abs(baseline_score)) * 100
            print(f"  ✅ DQN wins by {improvement:.1f}%")
        else:
            gap = ((baseline_score - dqn_score) / abs(baseline_score)) * 100
            print(f"  ⚠️  Baseline better by {gap:.1f}%")

    # Plot results
    plt.figure(figsize=(18, 5))

    # Plot 1: Training progress (smoothed)
    plt.subplot(1, 3, 1)
    plt.plot(training_rewards, alpha=0.2, color='blue', label='Raw')
    if len(training_rewards) > 20:
        smoothed = smooth_curve(training_rewards, window=20)
        plt.plot(range(10, 10 + len(smoothed)), smoothed, color='blue', linewidth=2, label='Smoothed (20)')
    plt.xlabel("Episode")
    plt.ylabel("Total Episode Reward")
    plt.title("DQN Training Progress (All Scenarios)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Per-scenario learning
    plt.subplot(1, 3, 2)
    colors = ['green', 'orange']
    for idx, (config_file, rewards) in enumerate(trainer.scenario_rewards.items()):
        scenario_name = eval_results[config_file]['name']
        episodes = [i * len(scenarios) + idx for i in range(len(rewards))]
        plt.plot(episodes, rewards, 'o-', alpha=0.5, color=colors[idx],
                 markersize=3, label=scenario_name)
    plt.xlabel("Episode")
    plt.ylabel("Total Episode Reward")
    plt.title("Learning per Scenario")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Final comparison
    plt.subplot(1, 3, 3)
    scenario_names = [eval_results[cf]['name'] for cf in scenarios]
    dqn_scores = [eval_results[cf]['per_step'] for cf in scenarios]
    baseline_scores = [baseline_results[cf]['per_step'] for cf in scenarios]

    x = np.arange(len(scenario_names))
    width = 0.35

    plt.bar(x - width/2, dqn_scores, width, label='DQN', color='green', alpha=0.7)
    plt.bar(x + width/2, baseline_scores, width, label='Fixed Baseline', color='blue', alpha=0.7)
    plt.xlabel("Scenario")
    plt.ylabel("Average Per-Step Reward")
    plt.title("DQN vs Baseline Comparison")
    plt.xticks(x, scenario_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("dqn_multi_scenario_results.png", dpi=150)
    print("\nPlot saved to dqn_multi_scenario_results.png")
    plt.show()

    # Overall verdict
    avg_dqn = np.mean([eval_results[cf]['per_step'] for cf in scenarios])
    avg_baseline = np.mean([baseline_results[cf]['per_step'] for cf in scenarios])

    print("\n" + "=" * 70)
    print("OVERALL PERFORMANCE")
    print("=" * 70)
    print(f"DQN Average:      {avg_dqn:.2f}")
    print(f"Baseline Average: {avg_baseline:.2f}")

    if avg_dqn > avg_baseline:
        improvement = ((avg_dqn - avg_baseline) / abs(avg_baseline)) * 100
        print(f"\n✅ SUCCESS! DQN beat the baseline by {improvement:.1f}% on average!")
    else:
        gap = ((avg_baseline - avg_dqn) / abs(avg_baseline)) * 100
        print(f"\n⚠️  DQN didn't beat baseline yet ({gap:.1f}% gap)")
        print("   Next steps: Try more episodes, lower learning rate, or enhanced architecture")
    print("=" * 70)

if __name__ == "__main__":
    train_and_evaluate()
