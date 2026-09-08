"""
test_env.py
Tests TraciWrapper and WaveGuardEnv with SUMO in headless mode.
"""

import pytest
import numpy as np
from pathlib import Path
from envs.waveguard_env import WaveGuardEnv

def test_waveguard_env_lifecycle():
    benchmark_cfg = Path(__file__).resolve().parent.parent / "sumo_nets" / "benchmark" / "benchmark_grid.sumocfg"
    assert benchmark_cfg.exists(), "Benchmark sumocfg should exist"

    env = WaveGuardEnv(sumocfg_path=benchmark_cfg, gui=False)
    obs, info = env.reset(seed=42)

    # Check TLS IDs initialized
    assert len(env.tls_ids) == 4, f"Expected 4 intersections in benchmark grid, got {len(env.tls_ids)}"
    
    # Check observation structure for each intersection
    for tid in env.tls_ids:
        assert tid in obs
        data = obs[tid]
        assert "bev_grid" in data
        assert "scalar_feats" in data
        assert "pressure" in data
        assert "neighbor_pressures" in data

        # Check shapes
        assert data["bev_grid"].shape == (2, 12, 12)
        assert data["scalar_feats"].shape == (3,)
        assert isinstance(data["pressure"], float)

    # Step environment
    action_dict = {tid: 0 for tid in env.tls_ids}
    next_obs, rewards, terminated, truncated, info = env.step(action_dict)

    assert len(next_obs) == 4
    assert len(rewards) == 4
    for tid in env.tls_ids:
        assert isinstance(rewards[tid], float)
        assert not np.isnan(rewards[tid])

    # Check metrics
    assert "overall_avg_wait_sec" in info
    assert "throughput_vehicles" in info
    assert "ttc_conflicts" in info

    env.close()
