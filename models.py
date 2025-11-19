import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Ensure CityFlow can be imported if installed locally
sys.path.insert(0, str(Path(__file__).resolve().parent / "CityFlow" / "build"))
import cityflow

class CityFlowEnv:
    def __init__(self, config_path, frame_skip=5):
        self.eng = cityflow.Engine(config_path, thread_num=1)
        self.intersection_id = self.eng.get_intersection_ids()[0]
        self.action_space = list(range(len(self.eng.get_intersection_phase(self.intersection_id))))
        self.frame_skip = frame_skip
        self.max_time = 3600
        self.reset()

    def reset(self):
        self.eng.reset()
        return self.get_state()

    def step(self, action):
        self.eng.set_tl_phase(self.intersection_id, action)
        for _ in range(self.frame_skip):
            self.eng.next_step()
        waiting_counts = list(self.eng.get_lane_waiting_vehicle_count().values())
        reward = -np.mean(waiting_counts) if waiting_counts else 0.0
        done = self.eng.get_current_time() >= self.max_time
        return self.get_state(), reward, done, {}

    def get_state(self):
        lane_counts = list(self.eng.get_lane_waiting_vehicle_count().values())
        max_count = max(lane_counts) if lane_counts else 1
        return np.array(lane_counts, dtype=np.float32) / max_count

class DQN(nn.Module):
    """Baseline DQN for Single-Agent."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)


class EnhancedDQN(nn.Module):
    """Enhanced DQN with deeper layers + normalization."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, size=5000):
        self.buffer = deque(maxlen=size)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class SingleAgentBaseline:
    def __init__(self, env: CityFlowEnv):
        self.env = env
        self.model = DQN(len(env.get_state()), len(env.action_space))
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()

    def train(self, episodes=30, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, decay=0.995):
        epsilon = epsilon_start
        rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            total = 0
            done = False
            while not done:
                if random.random() < epsilon:
                    action = random.choice(self.env.action_space)
                else:
                    with torch.no_grad():
                        q_vals = self.model(torch.tensor(state).float().unsqueeze(0))
                        action = q_vals.argmax().item()

                next_state, reward, done, _ = self.env.step(action)
                print(next_state)
                self.buffer.add((state, action, reward, next_state, done))
                state = next_state
                total += reward

                if len(self.buffer) > 64:
                    self._update(gamma)

            epsilon = max(epsilon_end, epsilon * decay)
            rewards.append(total)
            print(f"[SingleAgent] Ep {ep+1}: Reward={total:.2f}, Epsilon={epsilon:.3f}")
        return rewards

    def _update(self, gamma):
        batch = self.buffer.sample(64)
        states, actions, rewards_, next_states, dones = zip(*batch)
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).long()
        rewards_ = torch.tensor(rewards_).float()
        next_states = torch.tensor(next_states).float()
        dones = torch.tensor(dones).float()

        q_vals = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        q_next = self.model(next_states).max(1)[0].detach()
        q_target = rewards_ + gamma * q_next * (1 - dones)

        loss = nn.MSELoss()(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate(self, episodes=5):
        self.model.eval()
        rewards = []
        with torch.no_grad():
            for ep in range(episodes):
                state = self.env.reset()
                total = 0
                done = False
                while not done:
                    q_vals = self.model(torch.tensor(state).float().unsqueeze(0))
                    action = q_vals.argmax().item()
                    state, reward, done, _ = self.env.step(action)
                    total += reward
                rewards.append(total)
                print(f"[SingleAgent Eval] Ep {ep+1}: Reward={total:.2f}")
        return rewards

class EnhancedSingleAgent(SingleAgentBaseline):
    """Same as SingleAgent but uses EnhancedDQN."""
    def __init__(self, env: CityFlowEnv):
        super().__init__(env)
        self.model = EnhancedDQN(len(env.get_state()), len(env.action_space))
        self.optimizer = optim.Adam(self.model.parameters(), lr=8e-4)  # Slightly lower LR for stability

class MultiAgentDQN:
    def __init__(self, env: CityFlowEnv):
        self.env = env
        self.models = {iid: DQN(len(env.get_state()), len(env.action_space)) for iid in env.eng.get_intersection_ids()}
        self.buffers = {iid: deque(maxlen=5000) for iid in self.models}
        self.optimizers = {iid: optim.Adam(self.models[iid].parameters(), lr=1e-3) for iid in self.models}

    def train(self, episodes=30):
        rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            total = 0
            done = False
            while not done:
                for iid in self.models:
                    action = random.choice(self.env.action_space)
                    next_state, reward, done, _ = self.env.step(action)
                    self.buffers[iid].append((state, action, reward, next_state, done))
                    state = next_state
                    total += reward
                    if len(self.buffers[iid]) >= 64:
                        self._update(iid)
            print(f"[MultiAgent] Ep {ep+1}: Reward={total:.2f}")
            rewards.append(total)
        return rewards

    def _update(self, iid):
        batch = random.sample(self.buffers[iid], 64)
        states, actions, rewards_, next_states, dones = zip(*batch)
        model = self.models[iid]
        optimizer = self.optimizers[iid]

        states = torch.tensor(states).float()
        actions = torch.tensor(actions).long()
        rewards_ = torch.tensor(rewards_).float()
        next_states = torch.tensor(next_states).float()
        dones = torch.tensor(dones).float()

        q_vals = model(states).gather(1, actions.unsqueeze(1)).squeeze()
        q_next = model(next_states).max(1)[0].detach()
        q_target = rewards_ + 0.99 * q_next * (1 - dones)

        loss = nn.MSELoss()(q_vals, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()