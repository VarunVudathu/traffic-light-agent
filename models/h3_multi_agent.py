"""
H3: Multi-Agent DQN with Shared-Phase Coordination.

Hypothesis: Explicit sharing of immediate neighbor phases is sufficient for
coordination. Simple Shared-Phase DQN will match complex models without needing
Graph Attention Networks.

We implement two variants:
1. H3-Independent: Each agent trained independently, no coordination
2. H3-Shared-Phase: Agents observe neighbor current phases
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CityFlow" / "build"))
import cityflow


class MultiIntersectionEnv:
    """
    Environment for multi-agent traffic light control.

    Supports both independent and coordinated agents.
    """

    def __init__(self, config_path, frame_skip=1, max_steps=1000, coordination_mode='independent'):
        """
        Args:
            config_path: CityFlow config
            frame_skip: Steps to skip between actions
            max_steps: Maximum steps per episode
            coordination_mode: 'independent' or 'shared_phase'
        """
        self.eng = cityflow.Engine(config_path, thread_num=1)
        self.intersection_ids = self.eng.get_intersection_ids()
        self.num_agents = len(self.intersection_ids)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.current_step = 0
        self.coordination_mode = coordination_mode

        # Get action space for each intersection
        self.action_spaces = {}
        for iid in self.intersection_ids:
            phases = self.eng.get_intersection_phase(iid)
            self.action_spaces[iid] = list(range(len(phases)))

        # Build neighbor graph (for grid topology)
        self._build_neighbor_graph()

        # Track current phases for coordination
        self.current_phases = {iid: 0 for iid in self.intersection_ids}

        self.reset()

    def _build_neighbor_graph(self):
        """
        Build neighbor adjacency list.

        For grid topology (e.g., 2x2 or 3x3), neighbors are typically
        determined by road connections. Simplified here.
        """
        self.neighbors = {iid: [] for iid in self.intersection_ids}

        # If there's only 1 intersection, no neighbors
        if len(self.intersection_ids) == 1:
            return

        # For multi-intersection, assume grid topology
        # This is simplified - in practice, parse roadnet.json
        # For now, make each intersection connected to all others
        for iid in self.intersection_ids:
            self.neighbors[iid] = [other for other in self.intersection_ids if other != iid]

    def reset(self):
        """Reset environment."""
        self.eng.reset()
        self.current_step = 0
        self.current_phases = {iid: 0 for iid in self.intersection_ids}

        # Return initial states for all agents
        return {iid: self._get_state(iid) for iid in self.intersection_ids}

    def step(self, actions):
        """
        Take a step with actions from all agents.

        Args:
            actions: dict {intersection_id: action}

        Returns:
            states: dict of next states
            rewards: dict of rewards
            done: boolean
            info: dict
        """
        # Apply actions
        for iid, action in actions.items():
            self.eng.set_tl_phase(iid, action)
            self.current_phases[iid] = action

        # Step simulation
        for _ in range(self.frame_skip):
            self.eng.next_step()
        self.current_step += 1

        # Compute rewards
        rewards = {iid: self._compute_reward(iid) for iid in self.intersection_ids}

        # Get next states
        states = {iid: self._get_state(iid) for iid in self.intersection_ids}

        done = self.current_step >= self.max_steps

        return states, rewards, done, {}

    def _compute_reward(self, intersection_id):
        """Compute reward for a specific intersection."""
        # Use local waiting vehicles as reward
        lane_counts = list(self.eng.get_lane_waiting_vehicle_count().values())
        reward = -np.mean(lane_counts) if lane_counts else 0.0
        return reward

    def _get_state(self, intersection_id):
        """
        Get state for a specific intersection.

        State depends on coordination_mode:
        - 'independent': Only own queue state
        - 'shared_phase': Own queue state + neighbor phases
        """
        # Get own queue state
        lane_counts = list(self.eng.get_lane_waiting_vehicle_count().values())
        if not lane_counts:
            lane_counts = [0.0] * 12

        lane_counts = np.array(lane_counts, dtype=np.float32)
        max_count = max(lane_counts) if max(lane_counts) > 0 else 1.0
        own_state = lane_counts / max_count

        if self.coordination_mode == 'independent':
            return own_state

        elif self.coordination_mode == 'shared_phase':
            # Add neighbor phase information
            neighbor_phases = []
            for neighbor_id in self.neighbors[intersection_id]:
                # One-hot encode neighbor's current phase
                num_phases = len(self.action_spaces[neighbor_id])
                phase_one_hot = np.zeros(num_phases, dtype=np.float32)
                phase_one_hot[self.current_phases[neighbor_id]] = 1.0
                neighbor_phases.extend(phase_one_hot)

            # If no neighbors, add zeros
            if not neighbor_phases:
                neighbor_phases = [0.0]

            # Concatenate own state + neighbor phases
            full_state = np.concatenate([own_state, np.array(neighbor_phases)])
            return full_state

        else:
            raise ValueError(f"Unknown coordination_mode: {self.coordination_mode}")


class DQN(nn.Module):
    """DQN for multi-agent."""

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
    """Replay buffer."""

    def __init__(self, size=5000):
        self.buffer = deque(maxlen=size)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class H3MultiAgent:
    """
    Multi-Agent DQN system.

    Each intersection has its own DQN agent.
    Coordination mode determines if agents share information.
    """

    def __init__(self, env: MultiIntersectionEnv, learning_rate=1e-3):
        self.env = env
        self.intersection_ids = env.intersection_ids

        # Create a DQN and buffer for each agent
        self.models = {}
        self.optimizers = {}
        self.buffers = {}

        for iid in self.intersection_ids:
            # Get state size for this agent
            sample_state = env._get_state(iid)
            state_size = len(sample_state)
            action_size = len(env.action_spaces[iid])

            self.models[iid] = DQN(state_size, action_size)
            self.optimizers[iid] = optim.Adam(self.models[iid].parameters(), lr=learning_rate)
            self.buffers[iid] = ReplayBuffer()

    def train(self, episodes=50, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, decay=0.995, batch_size=128):
        """Train all agents."""
        epsilon = epsilon_start
        episode_rewards = []

        for ep in range(episodes):
            states = self.env.reset()
            total_reward = {iid: 0 for iid in self.intersection_ids}
            done = False

            while not done:
                # Get actions from all agents
                actions = {}
                for iid in self.intersection_ids:
                    if random.random() < epsilon:
                        actions[iid] = random.choice(self.env.action_spaces[iid])
                    else:
                        with torch.no_grad():
                            state_tensor = torch.tensor(states[iid]).float().unsqueeze(0)
                            q_vals = self.models[iid](state_tensor)
                            actions[iid] = q_vals.argmax().item()

                # Step environment
                next_states, rewards, done, _ = self.env.step(actions)

                # Store experiences and update
                for iid in self.intersection_ids:
                    self.buffers[iid].add((states[iid], actions[iid], rewards[iid], next_states[iid], done))
                    total_reward[iid] += rewards[iid]

                    # Update if enough data
                    if len(self.buffers[iid]) > batch_size:
                        self._update(iid, gamma, batch_size)

                states = next_states

            epsilon = max(epsilon_end, epsilon * decay)

            # Aggregate reward across all agents
            system_reward = sum(total_reward.values())
            episode_rewards.append(system_reward)

            if (ep + 1) % 10 == 0:
                avg = np.mean(episode_rewards[-10:])
                print(f"[H3-MultiAgent] Ep {ep+1}/{episodes}: System Reward={system_reward:.2f}, Avg(10)={avg:.2f}, ε={epsilon:.3f}")

        return episode_rewards

    def _update(self, iid, gamma, batch_size):
        """Update a specific agent's Q-network."""
        batch = self.buffers[iid].sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        next_states = torch.tensor(np.array(next_states)).float()
        dones = torch.tensor(dones).float()

        model = self.models[iid]
        optimizer = self.optimizers[iid]

        q_vals = model(states).gather(1, actions.unsqueeze(1)).squeeze()
        q_next = model(next_states).max(1)[0].detach()
        q_target = rewards + gamma * q_next * (1 - dones)

        loss = nn.MSELoss()(q_vals, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def evaluate(self, episodes=5):
        """Evaluate all agents."""
        for model in self.models.values():
            model.eval()

        episode_rewards = []

        with torch.no_grad():
            for ep in range(episodes):
                states = self.env.reset()
                total_reward = 0
                done = False

                while not done:
                    actions = {}
                    for iid in self.intersection_ids:
                        state_tensor = torch.tensor(states[iid]).float().unsqueeze(0)
                        q_vals = self.models[iid](state_tensor)
                        actions[iid] = q_vals.argmax().item()

                    states, rewards, done, _ = self.env.step(actions)
                    total_reward += sum(rewards.values())

                episode_rewards.append(total_reward)

        for model in self.models.values():
            model.train()

        return episode_rewards

    def save(self, path_prefix):
        """Save all agent models."""
        for iid, model in self.models.items():
            torch.save(model.state_dict(), f"{path_prefix}_{iid}.pth")

    def load(self, path_prefix):
        """Load all agent models."""
        for iid, model in self.models.items():
            model.load_state_dict(torch.load(f"{path_prefix}_{iid}.pth"))
