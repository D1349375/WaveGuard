"""
waveguard_env.py
Gymnasium-compatible Multi-Agent Environment for WaveGuard.
Extracts dual-channel Grid BEV matrices (Passenger cars vs Motorcycles),
models Taiwan traffic light action space, and computes the multi-objective reward:
R = alpha * R_flow + beta * R_safety + gamma * R_equity
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
from pathlib import Path
from .traci_wrapper import TraciWrapper

class RunningMeanStd:
    """Tracks running mean and variance for reward normalization (Section 2.4)."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, x):
        x = np.atleast_1d(np.asarray(x, 'float32'))
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-6)


class WaveGuardEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config_path=None, sumocfg_path=None, gui=False):
        super().__init__()

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "default_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        if sumocfg_path is None:
            sumocfg_path = Path(__file__).resolve().parent.parent / "sumo_nets" / "taiwan_xinyi" / "xinyi_corridor.sumocfg"
        
        self.wrapper = TraciWrapper(
            sumocfg_path=sumocfg_path,
            gui=gui,
            step_length=self.cfg["environment"]["sim_step"],
            lateral_resolution=self.cfg["environment"]["lateral_resolution"],
            ttc_threshold=self.cfg["reward_weights"]["ttc_threshold"]
        )

        self.grid_size = self.cfg["environment"]["grid_size"]
        self.grid_range = self.cfg["environment"]["grid_range"]
        self.alpha = self.cfg["reward_weights"]["alpha"]
        self.beta = self.cfg["reward_weights"]["beta"]
        self.gamma = self.cfg["reward_weights"]["gamma"]

        # Reward normalizers
        self.norm_flow = RunningMeanStd()
        self.norm_safety = RunningMeanStd()
        self.norm_equity = RunningMeanStd()

        self.prev_ttc_conflicts = 0
        self.tls_ids = []
        self.adj_matrix = None
        self.current_step = 0
        self.max_steps = 3600

    def reset(self, seed=42, options=None):
        super().reset(seed=seed)
        self.tls_ids = self.wrapper.start(seed=seed)
        self.current_step = 0
        self.prev_ttc_conflicts = 0
        
        # Build adjacency matrix for graph attention
        num_agents = len(self.tls_ids)
        self.adj_matrix = np.eye(num_agents, dtype=np.float32)
        
        # Connect linear or nearby neighbors
        for i in range(num_agents):
            for j in range(num_agents):
                if abs(i - j) == 1:
                    self.adj_matrix[i, j] = 1.0

        obs = self._get_observations()
        info = self.wrapper.get_aggregate_metrics()
        return obs, info

    def step(self, action_dict):
        """
        Takes green phase actions for each agent.
        Automatic safety transitions (Yellow + All-Red) are executed by wrapper.
        """
        # Command actions
        for tid, action in action_dict.items():
            self.wrapper.execute_phase(
                tls_id=tid,
                target_phase_idx=action,
                yellow_time=self.cfg["environment"]["yellow_time"],
                all_red_time=self.cfg["environment"]["all_red_time"]
            )

        # Advance simulation
        self.wrapper.step()
        self.current_step += 1

        obs = self._get_observations()
        rewards = self._compute_rewards()

        terminated = self.current_step >= self.max_steps
        truncated = False
        info = self.wrapper.get_aggregate_metrics()

        return obs, rewards, terminated, truncated, info

    def _get_observations(self):
        """
        Generates structured multi-agent observations:
        {tls_id: {'bev_grid': (2, N, N), 'scalar_feats': (3,), 'pressure': float, 'neighbor_pressures': np.ndarray}}
        """
        obs = {}
        pressures = [self.wrapper.get_intersection_pressure(tid) for tid in self.tls_ids]
        p_arr = np.array(pressures, dtype=np.float32)

        for i, tid in enumerate(self.tls_ids):
            car_g, moto_g, scalars = self.wrapper.get_grid_bev(
                tls_id=tid,
                grid_size=self.grid_size,
                grid_range=self.grid_range
            )
            # Stack into (2, N, N)
            bev_grid = np.stack([car_g, moto_g], axis=0)

            # Neighbor pressures aggregated via graph connectivity
            neighbors = np.dot(self.adj_matrix[i], p_arr)

            obs[tid] = {
                "bev_grid": bev_grid,
                "scalar_feats": scalars,
                "pressure": float(pressures[i]),
                "neighbor_pressures": neighbors,
                "movement_pressure": self.wrapper.get_movement_pressures(tid)
            }

        return obs

    def _compute_rewards(self):
        """
        Implements Section 2.4 Multi-Objective Reward:
        R = alpha * R_flow + beta * R_safety + gamma * R_equity
        """
        rewards = {}
        curr_ttc = self.wrapper.ttc_conflict_count
        delta_ttc = curr_ttc - self.prev_ttc_conflicts
        self.prev_ttc_conflicts = curr_ttc

        metrics = self.wrapper.get_aggregate_metrics()
        equity_delta = metrics["equity_wait_delta"]

        for tid in self.tls_ids:
            local_queue = self.wrapper.get_tls_halting(tid)
            r_flow = -float(local_queue)
            
            # R_safety: penalize TTC < 3.0s safety conflicts
            r_safety = -float(delta_ttc * 2.0)

            # R_equity: penalize wait-time gap between motorcycle and car
            r_equity = -float(equity_delta)

            # Update normalizers
            self.norm_flow.update(r_flow)
            self.norm_safety.update(r_safety)
            self.norm_equity.update(r_equity)

            # Normalized multi-objective sum
            r_flow_norm = self.norm_flow.normalize(r_flow)
            r_safety_norm = self.norm_safety.normalize(r_safety)
            r_equity_norm = self.norm_equity.normalize(r_equity)

            total_r = (self.alpha * r_flow_norm + 
                       self.beta * r_safety_norm + 
                       self.gamma * r_equity_norm)

            rewards[tid] = float(total_r)

        return rewards

    def close(self):
        self.wrapper.close()
