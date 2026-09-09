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
        batch_size=64,
        buffer_size=10000,
        device="cpu"
    ):
        self.tls_ids = tls_ids
        self.num_agents = len(tls_ids)
        self.id_to_idx = {tid: i for i, tid in enumerate(tls_ids)}
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
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

        # Minimum green time enforcement to avoid signal fluttering (Section 2.2)
        self.min_green = 10
        self.active_phase = {tid: 0 for tid in tls_ids}
        self.phase_timer = {tid: 0 for tid in tls_ids}

        self.train_step_count = 0

    def reset_timers(self):
        """Resets phase timers for a new episode."""
        self.active_phase = {tid: 0 for tid in self.tls_ids}
        self.phase_timer = {tid: 0 for tid in self.tls_ids}

    def decay_epsilon_for_episode(self, completed_episodes, total_episodes):
        """
        Sets epsilon for the upcoming episode, decaying smoothly from
        epsilon_start to epsilon_min over the first ~80% of total_episodes
        and holding at epsilon_min for the rest.

        See docs/rl_training_pitfalls_checklist.md item 1: the previous
        implementation decayed epsilon once per *training step* (inside
        train_step()) rather than once per episode. Combined with this
        environment's ~300 steps/episode and a 64-transition replay warm-up,
        that collapsed epsilon to its floor within the first episode --
        over 99% of a 25-episode run then trained with almost no exploration
        beyond whatever policy the agent stumbled into in the first ~100
        steps. Deriving the per-episode value directly from
        (completed_episodes, total_episodes) instead of a fixed multiplicative
        constant also means the schedule automatically rescales if the
        episode count changes, instead of needing the decay constant
        re-tuned by hand every time (a likely source of the original bug).
        """
        anchor_episodes = max(1, int(0.8 * total_episodes))
        progress = min(completed_episodes / anchor_episodes, 1.0)
        self.epsilon = self.epsilon_start * (self.epsilon_min / self.epsilon_start) ** progress

    def act(self, obs_dict, explore=True):
        """
        Takes decentralized actions for each intersection using GAT features.
        Enforces minimum green time (10s) to prevent signal fluttering.
        """
        actions = {}
        for tid in self.tls_ids:
            self.phase_timer[tid] += 1

        # Check which agents are eligible to change phase
        eligible = [tid for tid in self.tls_ids if self.phase_timer[tid] >= self.min_green]

        if not eligible:
            return {tid: self.active_phase[tid] for tid in self.tls_ids}

        # Exploration check
        if explore and random.random() < self.epsilon:
            for tid in eligible:
                new_act = random.randint(0, self.action_dim - 1)
                if new_act != self.active_phase[tid]:
                    self.active_phase[tid] = new_act
                    self.phase_timer[tid] = 0
            return {tid: self.active_phase[tid] for tid in self.tls_ids}

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

        for i, tid in enumerate(self.tls_ids):
            if tid in eligible:
                # Local directional pressure from BEV grid
                bev = obs_dict[tid]["bev_grid"]
                ew_p = float(np.sum(bev[:, :, 4:8]))
                ns_p = float(np.sum(bev[:, 4:8, :]))
                p_bias = np.array([ew_p - ns_p, ns_p - ew_p, 0.0, 0.0], dtype=np.float32)

                # GAT attention Q-values fused with pressure guidance (PressLight/CoLight formulation)
                q_score = q_vals[i].cpu().numpy() + 0.3 * p_bias

                # Taiwan-specific feature: if motorcycle stop box is saturated (>80%), prioritize arterial green
                stop_box_sat = obs_dict[tid]["scalar_feats"][0]
                if stop_box_sat > 0.8:
                    target_act = 0
                else:
                    target_act = int(np.argmax(q_score[:2]))

                if target_act != self.active_phase[tid]:
                    self.active_phase[tid] = target_act
                    self.phase_timer[tid] = 0

            actions[tid] = self.active_phase[tid]

        return actions

    def store_transition(self, obs, actions, rewards, next_obs, done):
        """Stores structured multi-agent transition in replay buffer."""
        self.memory.push(obs, actions, rewards, next_obs, done)

    def train_step(self):
        """Performs one vectorized step of Q-learning optimization."""
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        self.q_net.train()

        # Vectorized batch construction
        batch_bevs = []
        batch_scalars = []
        batch_actions = []
        batch_rewards = []
        batch_next_bevs = []
        batch_next_scalars = []
        batch_dones = []

        for obs_dict, action_dict, reward_dict, next_obs_dict, done in batch:
            batch_bevs.append([obs_dict[tid]["bev_grid"] for tid in self.tls_ids])
            batch_scalars.append([obs_dict[tid]["scalar_feats"] for tid in self.tls_ids])
            batch_actions.append([action_dict[tid] for tid in self.tls_ids])
            batch_rewards.append([reward_dict[tid] for tid in self.tls_ids])
            batch_next_bevs.append([next_obs_dict[tid]["bev_grid"] for tid in self.tls_ids])
            batch_next_scalars.append([next_obs_dict[tid]["scalar_feats"] for tid in self.tls_ids])
            batch_dones.append([1.0 if done else 0.0])

        b_bevs = torch.tensor(np.array(batch_bevs), dtype=torch.float32, device=self.device)          # (B, N, 2, G, G)
        b_scalars = torch.tensor(np.array(batch_scalars), dtype=torch.float32, device=self.device)    # (B, N, 3)
        b_actions = torch.tensor(np.array(batch_actions), dtype=torch.int64, device=self.device).unsqueeze(-1)  # (B, N, 1)
        b_rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32, device=self.device).unsqueeze(-1) # (B, N, 1)
        b_next_bevs = torch.tensor(np.array(batch_next_bevs), dtype=torch.float32, device=self.device)
        b_next_scalars = torch.tensor(np.array(batch_next_scalars), dtype=torch.float32, device=self.device)
        b_dones = torch.tensor(np.array(batch_dones), dtype=torch.float32, device=self.device).unsqueeze(-1) # (B, 1, 1)

        # Q prediction: (B, N, 1)
        q_preds = self.q_net(b_bevs, b_scalars, self.adj).gather(-1, b_actions)

        # Target Q: (B, N, 1)
        with torch.no_grad():
            next_q = self.target_net(b_next_bevs, b_next_scalars, self.adj).max(-1, keepdim=True)[0]
            q_targets = b_rewards + self.gamma * next_q * (1.0 - b_dones)

        loss = self.loss_fn(q_preds, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.train_step_count += 1

        # Target network update
        if self.train_step_count % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

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
