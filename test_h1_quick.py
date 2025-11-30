"""Quick test to verify H1 models work."""

from models.h1_basic import CityFlowEnv, H1BasicAgent
from models.baselines import FixedTimeBaseline
from pathlib import Path

# Test Fixed-Time baseline
print("Testing Fixed-Time baseline...")
config_path = str(Path("scenarios/configs/config_low_variance.json"))
baseline = FixedTimeBaseline(config_path, phase_duration=30, max_steps=100)  # Short run
rewards = baseline.run()
print(f"✅ Fixed-Time works! Mean reward: {sum(rewards)/len(rewards):.2f}\n")

# Test H1-Basic
print("Testing H1-Basic...")
env = CityFlowEnv(config_path, frame_skip=1, max_steps=100)
agent = H1BasicAgent(env)
training_rewards = agent.train(episodes=3)  # Just 3 episodes
print(f"✅ H1-Basic works! Final reward: {training_rewards[-1]:.2f}\n")

print("All models working correctly! ✅")
