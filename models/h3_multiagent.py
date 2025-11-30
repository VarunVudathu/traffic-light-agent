"""
H3-MultiAgent: Independent DQN agents per intersection.

Hypothesis: Multi-agent coordination using independent agents will improve scalability
and reduce computational overhead compared to single-agent approaches.

Key Features:
- Each intersection has its own DQN agent with local state/action space
- Independent learning (no explicit communication)
- Scalable to large networks (O(n) instead of O(n²) state space)
- Reduced training complexity per agent
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
from typing import Dict, List, Tuple

# Ensure CityFlow can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CityFlow" / "build"))
import cityflow


class MultiAgentEnv:
    """Multi-intersection traffic environment with independent agents."""

    def __init__(self, config_path, frame_skip=1, max_steps=1000):
        self.eng = cityflow.Engine(config_path, thread_num=1)
        self.intersection_ids = self.eng.get_intersection_ids()
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.current_step = 0

        # Store action spaces per intersection
        self.action_spaces = {}
        for iid in self.intersection_ids:
            phases = self.eng.get_intersection_phase(iid)
            self.action_spaces[iid] = list(range(len(phases)))

        # Map lanes to intersections for local state extraction
        self._map_lanes_to_intersections()

        self.reset()

    def _map_lanes_to_intersections(self):
        """
        Map lanes to their corresponding intersections.

        For simplicity, we use a heuristic: lanes are distributed uniformly
        across intersections based on lane naming/ordering.
        """
        self.intersection_lanes = {}
        all_lanes = sorted(self.eng.get_lane_waiting_vehicle_count().keys())

        if not all_lanes:
            # Fallback: assign empty lane lists
            for iid in self.intersection_ids:
                self.intersection_lanes[iid] = []
            return

        # Heuristic: distribute lanes evenly across intersections
        # This is a simplified assumption - in practice, would parse roadnet file
        lanes_per_intersection = max(1, len(all_lanes) // len(self.intersection_ids))

        for idx, iid in enumerate(self.intersection_ids):
            start = idx * lanes_per_intersection
            end = start + lanes_per_intersection if idx < len(self.intersection_ids) - 1 else len(all_lanes)
            self.intersection_lanes[iid] = all_lanes[start:end]

    def reset(self):
        """Reset environment and return initial states for all agents."""
        self.eng.reset()
        self.current_step = 0
        return self.get_states()

    def get_states(self) -> Dict[str, np.ndarray]:
        """
        Get local state for each intersection.

        Returns:
            Dict mapping intersection_id -> state vector
        """
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        states = {}

        for iid in self.intersection_ids:
            # Get waiting counts for lanes belonging to this intersection
            local_lanes = self.intersection_lanes.get(iid, [])
            local_counts = [lane_waiting.get(lane, 0) for lane in local_lanes]

            if not local_counts or all(c == 0 for c in local_counts):
                # If no data or all zeros, return normalized zero vector
                states[iid] = np.zeros(len(local_lanes) if local_lanes else 1, dtype=np.float32)
            else:
                # Normalize by max local count
                max_count = max(local_counts)
                states[iid] = np.array(local_counts, dtype=np.float32) / max_count

        return states

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Execute actions for all agents.

        Args:
            actions: Dict mapping intersection_id -> action

        Returns:
            (states, rewards, done, info)
        """
        # Set traffic light phases for all intersections
        for iid, action in actions.items():
            self.eng.set_tl_phase(iid, action)

        # Simulate for frame_skip steps
        for _ in range(self.frame_skip):
            self.eng.next_step()

        self.current_step += 1

        # Compute rewards (local rewards per intersection)
        rewards = self._compute_rewards()

        # Check if done
        done = self.current_step >= self.max_steps

        # Get next states
        next_states = self.get_states()

        return next_states, rewards, done, {}

    def _compute_rewards(self) -> Dict[str, float]:
        """
        Compute local reward for each intersection.

        Uses negative mean waiting vehicles for lanes controlled by this intersection.
        """
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        rewards = {}

        for iid in self.intersection_ids:
            local_lanes = self.intersection_lanes.get(iid, [])
            local_waiting = [lane_waiting.get(lane, 0) for lane in local_lanes]

            # Reward is negative mean waiting (same as H1-Basic, but localized)
            rewards[iid] = -np.mean(local_waiting) if local_waiting else 0.0

        return rewards


class DQN(nn.Module):
    """DQN network for individual agent."""

    def __init__(self, input_size, output_size):
        super().__init__()
        # Smaller network since each agent has smaller local state space
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for individual agent."""

    def __init__(self, size=5000):
        self.buffer = deque(maxlen=size)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class IndependentAgent:
    """Independent DQN agent for one intersection."""

    def __init__(self, intersection_id: str, state_size: int, action_size: int, learning_rate=1e-3):
        self.intersection_id = intersection_id
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer()
        self.action_space = list(range(action_size))

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.choice(self.action_space)
        else:
            with torch.no_grad():
                q_vals = self.model(torch.tensor(state).float().unsqueeze(0))
                return q_vals.argmax().item()

    def update(self, gamma: float, batch_size: int):
        """Update Q-network using sampled batch."""
        if len(self.buffer) < batch_size:
            return

        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        next_states = torch.tensor(np.array(next_states)).float()
        dones = torch.tensor(dones).float()

        q_vals = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        q_next = self.model(next_states).max(1)[0].detach()
        q_target = rewards + gamma * q_next * (1 - dones)

        loss = nn.MSELoss()(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class H3MultiAgentSystem:
    """Coordinated system of independent DQN agents."""

    def __init__(self, env: MultiAgentEnv, learning_rate=1e-3):
        self.env = env
        self.agents = {}

        # Create independent agent for each intersection
        initial_states = env.get_states()
        for iid in env.intersection_ids:
            state_size = len(initial_states[iid])
            action_size = len(env.action_spaces[iid])
            self.agents[iid] = IndependentAgent(iid, state_size, action_size, learning_rate)

    def train(self, episodes=50, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, decay=0.995, batch_size=128):
        """Train all agents using independent learning."""
        epsilon = epsilon_start
        episode_rewards = []

        for ep in range(episodes):
            states = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                # Each agent selects action independently
                actions = {}
                for iid, agent in self.agents.items():
                    actions[iid] = agent.select_action(states[iid], epsilon)

                # Execute joint action
                next_states, rewards, done, _ = self.env.step(actions)

                # Store experiences and update each agent independently
                for iid, agent in self.agents.items():
                    agent.buffer.add((
                        states[iid],
                        actions[iid],
                        rewards[iid],
                        next_states[iid],
                        done
                    ))
                    agent.update(gamma, batch_size)

                states = next_states
                total_reward += sum(rewards.values())  # Sum of all agent rewards

            epsilon = max(epsilon_end, epsilon * decay)
            episode_rewards.append(total_reward)

            if (ep + 1) % 10 == 0:
                avg = np.mean(episode_rewards[-10:])
                print(f"[H3-MultiAgent] Ep {ep+1}/{episodes}: Reward={total_reward:.2f}, "
                      f"Avg(10)={avg:.2f}, ε={epsilon:.3f}, Agents={len(self.agents)}")

        return episode_rewards

    def evaluate(self, episodes=5):
        """Evaluate trained agents."""
        for agent in self.agents.values():
            agent.model.eval()

        episode_rewards = []

        with torch.no_grad():
            for ep in range(episodes):
                states = self.env.reset()
                total = 0
                done = False

                while not done:
                    # Greedy action selection
                    actions = {}
                    for iid, agent in self.agents.items():
                        q_vals = agent.model(torch.tensor(states[iid]).float().unsqueeze(0))
                        actions[iid] = q_vals.argmax().item()

                    states, rewards, done, _ = self.env.step(actions)
                    total += sum(rewards.values())

                episode_rewards.append(total)

        for agent in self.agents.values():
            agent.model.train()

        return episode_rewards

    def save(self, directory):
        """Save all agent models."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        for iid, agent in self.agents.items():
            path = Path(directory) / f"agent_{iid}.pth"
            torch.save(agent.model.state_dict(), path)

    def load(self, directory):
        """Load all agent models."""
        for iid, agent in self.agents.items():
            path = Path(directory) / f"agent_{iid}.pth"
            agent.model.load_state_dict(torch.load(path))
