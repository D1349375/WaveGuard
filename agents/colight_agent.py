"""
colight_agent.py
CoLight Multi-Agent Reinforcement Learning Agent for WaveGuard.
Adopts Centralized Training with Decentralized Execution (CTDE).
Learns multi-intersection coordination with Graph Attention Network (GAT)
over Grid BEV occupancy matrices.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path
from .gat_net import CoLightNetwork

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class WaveGuardCoLightAgent:
    def __init__(
        self,
        tls_ids,
        grid_size=12,
        scalar_dim=3,
        hidden_dim=64,
        num_heads=4,
        action_dim=4,
        lr=0.0005,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.985,
        batch_size=64,
        buffer_size=10000,
        device="cpu"
    ):
        self.tls_ids = tls_ids
        self.num_agents = len(tls_ids)
        self.id_to_idx = {tid: i for i, tid in enumerate(tls_ids)}
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Build Adjacency Matrix
        self.adj = torch.eye(self.num_agents, dtype=torch.float32, device=self.device)
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if abs(i - j) == 1:
                    self.adj[i, j] = 1.0

        # Networks
        self.q_net = CoLightNetwork(
            grid_size=grid_size,
            scalar_dim=scalar_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            action_dim=action_dim
        ).to(self.device)

        self.target_net = CoLightNetwork(
            grid_size=grid_size,
            scalar_dim=scalar_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            action_dim=action_dim
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability
        self.memory = ReplayBuffer(capacity=buffer_size)

        self.train_step_count = 0

    def act(self, obs_dict, explore=True):
        """
        Takes decentralized actions for each intersection using GAT features.
        """
        if explore and random.random() < self.epsilon:
            return {tid: random.randint(0, self.action_dim - 1) for tid in self.tls_ids}

        # Prepare batch tensors for all nodes: (num_agents, 2, N, N)
        bev_list = []
        scalar_list = []
        for tid in self.tls_ids:
            bev_list.append(obs_dict[tid]["bev_grid"])
            scalar_list.append(obs_dict[tid]["scalar_feats"])

        bevs = torch.tensor(np.array(bev_list), dtype=torch.float32, device=self.device)
        scalars = torch.tensor(np.array(scalar_list), dtype=torch.float32, device=self.device)

        self.q_net.eval()
        with torch.no_grad():
            q_vals = self.q_net(bevs, scalars, self.adj)  # (num_agents, action_dim)
            best_actions = torch.argmax(q_vals, dim=-1).cpu().numpy()

        return {tid: int(best_actions[i]) for i, tid in enumerate(self.tls_ids)}

    def store_transition(self, obs, actions, rewards, next_obs, done):
        """Stores structured multi-agent transition in replay buffer."""
        self.memory.push(obs, actions, rewards, next_obs, done)

    def train_step(self):
        """Performs one step of Q-learning optimization."""
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        losses = []

        self.q_net.train()

        # Batch preparation
        for obs_dict, action_dict, reward_dict, next_obs_dict, done in batch:
            bevs = torch.tensor(
                np.array([obs_dict[tid]["bev_grid"] for tid in self.tls_ids]),
                dtype=torch.float32, device=self.device
            )
            scalars = torch.tensor(
                np.array([obs_dict[tid]["scalar_feats"] for tid in self.tls_ids]),
                dtype=torch.float32, device=self.device
            )
            actions = torch.tensor(
                [action_dict[tid] for tid in self.tls_ids],
                dtype=torch.int64, device=self.device
            ).unsqueeze(-1)
            rewards = torch.tensor(
                [reward_dict[tid] for tid in self.tls_ids],
                dtype=torch.float32, device=self.device
            ).unsqueeze(-1)

            next_bevs = torch.tensor(
                np.array([next_obs_dict[tid]["bev_grid"] for tid in self.tls_ids]),
                dtype=torch.float32, device=self.device
            )
            next_scalars = torch.tensor(
                np.array([next_obs_dict[tid]["scalar_feats"] for tid in self.tls_ids]),
                dtype=torch.float32, device=self.device
            )

            # Current Q
            q_pred = self.q_net(bevs, scalars, self.adj).gather(1, actions)

            # Target Q (Double DQN or Bellman target)
            with torch.no_grad():
                if done:
                    q_target = rewards
                else:
                    next_q = self.target_net(next_bevs, next_scalars, self.adj).max(1, keepdim=True)[0]
                    q_target = rewards + self.gamma * next_q

            loss = self.loss_fn(q_pred, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
            self.optimizer.step()

            losses.append(loss.item())

        self.train_step_count += 1

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Target network update
        if self.train_step_count % 10 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(np.mean(losses))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }, path)
        print(f"Model successfully saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        print(f"Model successfully loaded from {path}")
