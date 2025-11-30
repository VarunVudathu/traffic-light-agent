"""
H2: Single-Agent DQN with MaxPressure Reward Decoupling.

Tests H2 hypothesis:
"Decoupling the reward signal from queue length to MaxPressure (incoming - outgoing)
will improve learning efficiency and convergence compared to standard queue-based rewards."

Key difference from H1-Basic:
- Reward function uses MaxPressure instead of average queue length
- Same network architecture and training procedure
- State representation remains queue-based (same as H1-Basic)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pathlib import Path

# Ensure CityFlow can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CityFlow" / "build"))
import cityflow

# Import MaxPressure utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.maxpressure import compute_maxpressure_reward


class CityFlowEnvMaxPressure:
    """Traffic simulation environment using CityFlow with MaxPressure reward."""

    def __init__(self, config_path, frame_skip=1, max_steps=1000):
        self.eng = cityflow.Engine(config_path, thread_num=1)
        self.intersection_id = self.eng.get_intersection_ids()[0]
        self.action_space = list(range(len(self.eng.get_intersection_phase(self.intersection_id))))
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.current_step = 0
        self.current_phase = 0
        self.reset()

    def reset(self):
        self.eng.reset()
        self.current_step = 0
        self.current_phase = 0
        return self.get_state()

    def step(self, action):
        self.current_phase = action
        self.eng.set_tl_phase(self.intersection_id, action)

        for _ in range(self.frame_skip):
            self.eng.next_step()

        self.current_step += 1

        # NEW: Use MaxPressure reward instead of average queue length
        reward = compute_maxpressure_reward(
            self.eng,
            self.intersection_id,
            self.current_phase
        )

        done = self.current_step >= self.max_steps

        return self.get_state(), reward, done, {}

    def get_state(self):
        """State: Normalized lane waiting vehicle counts (same as H1-Basic)."""
        lane_counts = list(self.eng.get_lane_waiting_vehicle_count().values())
        if not lane_counts:
            return np.array([0.0], dtype=np.float32)
        max_count = max(lane_counts)
        if max_count == 0:
            return np.array(lane_counts, dtype=np.float32)
        return np.array(lane_counts, dtype=np.float32) / max_count


class DQN(nn.Module):
    """Standard DQN network (same architecture as H1)."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, size=50000):
        self.buffer = deque(maxlen=size)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class H2MaxPressureAgent:
    """H2: Single-Agent DQN with MaxPressure reward."""

    def __init__(self, env: CityFlowEnvMaxPressure, learning_rate=1e-3):
        self.env = env
        self.model = DQN(len(env.get_state()), len(env.action_space))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer()

    def train(self, episodes=50, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, decay=0.995, batch_size=128):
        """Train the agent using MaxPressure reward."""
        epsilon = epsilon_start
        rewards = []

        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.choice(self.env.action_space)
                else:
                    with torch.no_grad():
                        q_vals = self.model(torch.tensor(state).float().unsqueeze(0))
                        action = q_vals.argmax().item()

                next_state, reward, done, _ = self.env.step(action)
                self.buffer.add((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                # Update model
                if len(self.buffer) > batch_size:
                    self._update(gamma, batch_size)

            epsilon = max(epsilon_end, epsilon * decay)
            rewards.append(total_reward)

            if (ep + 1) % 10 == 0:
                avg = np.mean(rewards[-10:])
                print(f"[H2-MaxPressure] Ep {ep+1}/{episodes}: Reward={total_reward:.2f}, Avg(10)={avg:.2f}, ε={epsilon:.3f}")

        return rewards

    def _update(self, gamma, batch_size):
        """Update Q-network using sampled batch."""
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        next_states = torch.tensor(np.array(next_states)).float()
        dones = torch.tensor(dones).float()

        # Q-learning update
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            targets = rewards + gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state, epsilon=0.0):
        """Select action (for evaluation)."""
        if random.random() < epsilon:
            return random.choice(self.env.action_space)
        with torch.no_grad():
            q_vals = self.model(torch.tensor(state).float().unsqueeze(0))
            return q_vals.argmax().item()
