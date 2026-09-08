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
    mock_obs = {
        tid: {
            "bev_grid": np.ones((2, 12, 12), dtype=np.float32),
            "scalar_feats": np.array([0.9, 0.1, 0.0], dtype=np.float32)
        } for tid in tls_ids
    }
    mp_actions = mp.act(mock_obs)
    assert "J0" in mp_actions

def test_failsafe_coordinator():
    tls_ids = ["J0", "J1"]
    agent = WaveGuardCoLightAgent(tls_ids=tls_ids, grid_size=12)
    mp = MaxPressureController(tls_ids)
    coord = FailSafeCoordinator(agent, mp)

    mock_obs = {
        tid: {
            "bev_grid": np.zeros((2, 12, 12), dtype=np.float32),
            "scalar_feats": np.array([0.0, 0.0, 0.0], dtype=np.float32)
        } for tid in tls_ids
    }

    # Normal act
    actions, mode = coord.act(mock_obs, force_degrade=False)
    assert mode == "MARL_COOPERATIVE"

    # Degraded act
    actions_deg, mode_deg = coord.act(mock_obs, force_degrade=True)
    assert mode_deg == "FAILSAFE_MAX_PRESSURE"
