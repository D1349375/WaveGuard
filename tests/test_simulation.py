"""
test_simulation.py
Tests the Taiwan Xinyi Corridor network and incident injection mechanism.
"""

import pytest
from pathlib import Path
from envs.waveguard_env import WaveGuardEnv
from agents.colight_agent import WaveGuardCoLightAgent
from agents.rule_controllers import FixedTimeController, MaxPressureController

def test_xinyi_corridor_simulation():
    xinyi_cfg = Path(__file__).resolve().parent.parent / "sumo_nets" / "taiwan_xinyi" / "xinyi_corridor.sumocfg"
    assert xinyi_cfg.exists(), "Xinyi sumocfg must exist"

    env = WaveGuardEnv(sumocfg_path=xinyi_cfg, gui=False)
    obs, info = env.reset(seed=123)

    assert len(env.tls_ids) == 6, f"Expected 6 intersections along corridor, got {len(env.tls_ids)}"

    # Run 15 simulation steps
    agent = WaveGuardCoLightAgent(tls_ids=env.tls_ids, grid_size=12, action_dim=4)
    for _ in range(15):
        actions = agent.act(obs, explore=True)
        obs, rewards, term, trunc, info = env.step(actions)

    assert info["step"] == 15
    assert info["throughput_vehicles"] >= 0

    # Test incident injection
    target_edge = "E_J1_J2"
    veh = env.wrapper.inject_incident(target_edge, speed=0.0, duration=10)
    # veh can be None if no vehicle on edge at step 15, or str
    env.close()
