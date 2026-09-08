"""
test_agent.py
Unit tests for neural network architectures, MARL agents, and rule controllers.
"""

import pytest
import torch
import numpy as np
from agents.gat_net import BEVEncoder, GraphAttentionLayer, CoLightNetwork
from agents.colight_agent import WaveGuardCoLightAgent
from agents.rule_controllers import FixedTimeController, MaxPressureController, FailSafeCoordinator

def test_bev_encoder_forward():
    batch_size = 4
    grid_size = 12
    encoder = BEVEncoder(grid_size=grid_size, scalar_dim=3, out_dim=64)
    
    bev_grid = torch.rand(batch_size, 2, grid_size, grid_size)
    scalar_feats = torch.rand(batch_size, 3)
    out = encoder(bev_grid, scalar_feats)
    
    assert out.shape == (batch_size, 64)
    assert not torch.isnan(out).any()

def test_gat_layer_forward():
    num_nodes = 6
    in_dim = 64
    gat = GraphAttentionLayer(in_features=in_dim, out_features=in_dim, num_heads=4)
    
    h = torch.rand(num_nodes, in_dim)
    adj = torch.eye(num_nodes)
    for i in range(num_nodes - 1):
        adj[i, i+1] = 1.0
        adj[i+1, i] = 1.0
        
    out = gat(h, adj)
    assert out.shape == (num_nodes, in_dim)
    assert not torch.isnan(out).any()

def test_gat_layer_actually_uses_neighbor_information():
    """
    A GAT layer that produces the right output *shape* regardless of the
    adjacency matrix would still pass test_gat_layer_forward -- this is the
    "structurally present but behaviorally inert" failure shape (e.g. a
    graph-attention layer that degrades to per-node processing because the
    mask/adjacency never actually reaches the attention weights). Here we
    change a neighbor's features and confirm node 0's output changes only
    when it is actually connected to that neighbor, and is unaffected when it
    is isolated -- proving messages really flow along the adjacency matrix
    rather than being ignored or leaking between unconnected nodes.
    """
    torch.manual_seed(0)
    num_nodes = 3
    in_dim = 16
    gat = GraphAttentionLayer(in_features=in_dim, out_features=in_dim, num_heads=4)
    gat.eval()

    h = torch.rand(num_nodes, in_dim)
    h_perturbed = h.clone()
    h_perturbed[2] = torch.rand(in_dim) * 10.0  # drastically change node 2

    adj_isolated = torch.eye(num_nodes)  # node 0 has no neighbors
    adj_connected = torch.eye(num_nodes)
    adj_connected[0, 2] = 1.0
    adj_connected[2, 0] = 1.0

    with torch.no_grad():
        out_isolated_before = gat(h, adj_isolated)[0]
        out_isolated_after = gat(h_perturbed, adj_isolated)[0]

        out_connected_before = gat(h, adj_connected)[0]
        out_connected_after = gat(h_perturbed, adj_connected)[0]

    # Node 0 is isolated in adj_isolated -> perturbing node 2 must not change it.
    assert torch.allclose(out_isolated_before, out_isolated_after, atol=1e-6)

    # Node 0 is connected to node 2 in adj_connected -> perturbing node 2 must
    # change node 0's aggregated output.
    assert not torch.allclose(out_connected_before, out_connected_after, atol=1e-4)

def test_colight_network_forward():
    num_nodes = 6
    grid_size = 12
    net = CoLightNetwork(grid_size=grid_size, scalar_dim=3, hidden_dim=64, num_heads=4, action_dim=4)
    
    bevs = torch.rand(num_nodes, 2, grid_size, grid_size)
    scalars = torch.rand(num_nodes, 3)
    adj = torch.eye(num_nodes)
    
    q_vals = net(bevs, scalars, adj)
    assert q_vals.shape == (num_nodes, 4)
    assert not torch.isnan(q_vals).any()

def test_colight_agent_act_and_store():
    tls_ids = ["J0", "J1", "J2"]
    agent = WaveGuardCoLightAgent(tls_ids=tls_ids, grid_size=12, action_dim=4)
    
    # Mock observation dictionary
    obs = {}
    for tid in tls_ids:
        obs[tid] = {
            "bev_grid": np.zeros((2, 12, 12), dtype=np.float32),
            "scalar_feats": np.array([0.5, 0.2, 0.0], dtype=np.float32),
            "pressure": 5.0,
            "neighbor_pressures": np.array([5.0, 5.0, 5.0])
        }
        
    actions = agent.act(obs, explore=False)
    assert len(actions) == len(tls_ids)
    for tid in tls_ids:
        assert 0 <= actions[tid] < 4

    rewards = {tid: 1.0 for tid in tls_ids}
    agent.store_transition(obs, actions, rewards, obs, False)
    assert len(agent.memory) == 1

def test_rule_controllers():
    tls_ids = ["J0", "J1"]
    fixed = FixedTimeController(tls_ids, phase_durations=[2, 2])
    act1 = fixed.act()
    act2 = fixed.act()
    act3 = fixed.act()
    assert act1["J0"] == 0
    assert act3["J0"] == 1  # Switched after duration

    mp = MaxPressureController(tls_ids, min_green=1)

    # Arterial pressure >> cross pressure and stop-box not saturated -> should pick phase 0
    mock_obs_arterial = {
        tid: {
            "bev_grid": np.ones((2, 12, 12), dtype=np.float32),
            "scalar_feats": np.array([0.1, 0.1, 0.0], dtype=np.float32),
            "movement_pressure": {"arterial": 20.0, "cross": 2.0}
        } for tid in tls_ids
    }
    mp_actions = mp.act(mock_obs_arterial)
    assert "J0" in mp_actions
    assert mp_actions["J0"] == 0

    # Cross pressure >> arterial pressure -> should pick phase 1 (behavioral check,
    # not just "a dict came back" — this would fail if the controller ignored the
    # movement pressure signal entirely)
    mp2 = MaxPressureController(tls_ids, min_green=1)
    mock_obs_cross = {
        tid: {
            "bev_grid": np.ones((2, 12, 12), dtype=np.float32),
            "scalar_feats": np.array([0.1, 0.1, 0.0], dtype=np.float32),
            "movement_pressure": {"arterial": 2.0, "cross": 20.0}
        } for tid in tls_ids
    }
    mp2_actions = mp2.act(mock_obs_cross)
    assert mp2_actions["J0"] == 1

    # Saturated motorcycle stop box overrides pressure and forces arterial clearance
    mp3 = MaxPressureController(tls_ids, min_green=1)
    mock_obs_saturated = {
        tid: {
            "bev_grid": np.ones((2, 12, 12), dtype=np.float32),
            "scalar_feats": np.array([0.9, 0.1, 0.0], dtype=np.float32),
            "movement_pressure": {"arterial": 2.0, "cross": 20.0}
        } for tid in tls_ids
    }
    mp3_actions = mp3.act(mock_obs_saturated)
    assert mp3_actions["J0"] == 0

def test_failsafe_coordinator():
    tls_ids = ["J0", "J1"]
    agent = WaveGuardCoLightAgent(tls_ids=tls_ids, grid_size=12)
    mp = MaxPressureController(tls_ids)
    coord = FailSafeCoordinator(agent, mp, heartbeat_timeout=3)

    mock_obs = {
        tid: {
            "bev_grid": np.zeros((2, 12, 12), dtype=np.float32),
            "scalar_feats": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "movement_pressure": {"arterial": 1.0, "cross": 1.0}
        } for tid in tls_ids
    }

    # Normal act
    actions, mode = coord.act(mock_obs, force_degrade=False)
    assert mode == "MARL_COOPERATIVE"

    # Manually forced degrade path: a single forced miss is tolerated (holds the
    # last known-good action) since heartbeat_timeout=3; sustained forcing past
    # the threshold is what actually triggers the rule-based fallback.
    for _ in range(coord.heartbeat_timeout):
        actions_deg, mode_deg = coord.act(mock_obs, force_degrade=True)
    assert mode_deg == "FAILSAFE_MAX_PRESSURE"


def test_failsafe_coordinator_auto_detects_real_sensor_failure(monkeypatch):
    """
    The watchdog's whole point (Section 2.5) is to notice the MARL agent's inputs
    are unhealthy on its own, not just to branch on a flag the caller already knows
    the answer to. This drives real exceptions through marl_agent.act (simulating a
    genuine sensor/comms fault) and checks:
    1. A short run of misses (below heartbeat_timeout) is tolerated.
    2. Sustained failure (>= heartbeat_timeout) actually degrades to Max Pressure.
    3. The coordinator recovers automatically once inputs are healthy again,
       rather than permanently latching into fallback mode (a real bug found in
       the original implementation: consecutive_misses reaching the timeout made
       sensor_healthy permanently False, since nothing ever probed the primary
       agent again to reset the counter).
    """
    tls_ids = ["J0", "J1"]
    agent = WaveGuardCoLightAgent(tls_ids=tls_ids, grid_size=12)
    mp = MaxPressureController(tls_ids)
    coord = FailSafeCoordinator(agent, mp, heartbeat_timeout=3)

    mock_obs = {
        tid: {
            "bev_grid": np.zeros((2, 12, 12), dtype=np.float32),
            "scalar_feats": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "movement_pressure": {"arterial": 1.0, "cross": 1.0}
        } for tid in tls_ids
    }

    # Establish a healthy baseline first, so there is a "last known-good" MARL
    # action for the coordinator to fall back on during brief blips.
    monkeypatch.setattr(agent, "act", lambda obs, explore=False: {tid: 0 for tid in tls_ids})
    _, mode0 = coord.act(mock_obs)
    assert mode0 == "MARL_COOPERATIVE"

    def broken_act(obs, explore=False):
        raise RuntimeError("simulated sensor timeout")

    monkeypatch.setattr(agent, "act", broken_act)

    # Brief blip: consecutive_misses stays below heartbeat_timeout, so the
    # coordinator should still report MARL_COOPERATIVE (holding the last-good
    # action) instead of dropping straight to the rule-based fallback.
    _, mode1 = coord.act(mock_obs)
    assert coord.consecutive_misses == 1
    assert mode1 == "MARL_COOPERATIVE"

    # Keep failing until the miss count reaches the timeout threshold.
    actions, mode = None, None
    for _ in range(5):
        actions, mode = coord.act(mock_obs)

    assert mode == "FAILSAFE_MAX_PRESSURE"
    assert coord.is_fallback_active is True
    assert set(actions.keys()) == set(tls_ids)

    # Once the sensor is healthy again, the coordinator must recover on its own
    # -- this is the exact scenario the pre-fix implementation could never
    # reach, because it stopped probing the primary agent once tripped.
    monkeypatch.setattr(agent, "act", lambda obs, explore=False: {tid: 1 for tid in tls_ids})
    actions_ok, mode_ok = coord.act(mock_obs)
    assert mode_ok == "MARL_COOPERATIVE"
    assert coord.is_fallback_active is False
    assert coord.consecutive_misses == 0
