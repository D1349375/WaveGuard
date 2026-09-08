"""
gat_net.py
Graph Attention Network (GAT) and BEV Convolutional Encoder for WaveGuard.
Implements the CoLight multi-agent cooperative attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVEncoder(nn.Module):
    """
    Encodes the (2, N, N) dual-channel BEV occupancy matrix (cars & motorcycles)
    plus the 3-dim scalar vector into a compact feature vector.
    """
    def __init__(self, grid_size=12, scalar_dim=3, out_dim=64):
        super().__init__()
        # 2 channels: channel 0 = cars, channel 1 = motorcycles
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, N/2, N/2)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3))  # (32, 3, 3) -> 288
        )
        conv_flat_dim = 32 * 3 * 3
        self.fc = nn.Sequential(
            nn.Linear(conv_flat_dim + scalar_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, bev_grid, scalar_feats):
        # bev_grid: (batch, 2, N, N)
        # scalar_feats: (batch, 3)
        b_feat = self.conv(bev_grid)
        b_flat = b_feat.view(b_feat.size(0), -1)
        combined = torch.cat([b_flat, scalar_feats], dim=-1)
        return self.fc(combined)


class GraphAttentionLayer(nn.Module):
    """
    Multi-Head Graph Attention Layer (CoLight style).
    Aggregates traffic state representations of adjacent intersections.
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        """
        h: (num_nodes, in_features) or (batch_size, num_nodes, in_features)
        adj: (num_nodes, num_nodes) adjacency matrix
        """
        is_batched = (h.dim() == 3)
        if not is_batched:
            h = h.unsqueeze(0)  # (1, num_nodes, in_features)

        batch_size, num_nodes, _ = h.shape
        Wh = self.W(h).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        attn_src = (Wh * self.a_src).sum(dim=-1)  # (batch_size, num_nodes, num_heads)
        attn_dst = (Wh * self.a_dst).sum(dim=-1)  # (batch_size, num_nodes, num_heads)

        # Broadcast for all pairs (i, j): (batch_size, num_nodes, num_nodes, num_heads)
        attn = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)
        attn = self.leaky_relu(attn)

        # Mask non-connected edges: (1, num_nodes, num_nodes, 1)
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)
        attn = attn.masked_fill(mask, -1e9)

        # Softmax over neighbors (dim=2)
        alpha = F.softmax(attn, dim=2)
        alpha = self.dropout(alpha)

        # Weighted sum across neighbors: alpha (B, N, M, H) * Wh (B, M, H, D) -> (B, N, H, D)
        # Using einsum for clean vectorized computation
        h_prime = torch.einsum("bnmh,bmhd->bnhd", alpha, Wh)
        h_prime = h_prime.contiguous().view(batch_size, num_nodes, self.out_features)
        out = F.elu(h_prime)

        if not is_batched:
            out = out.squeeze(0)
        return out


class CoLightNetwork(nn.Module):
    """
    Complete CoLight MARL Network:
    BEVEncoder -> GAT Message Passing -> Action-Value (Q-Network) Head.
    Supports both single-step and batched inference/training.
    """
    def __init__(self, grid_size=12, scalar_dim=3, hidden_dim=64, num_heads=4, action_dim=4):
        super().__init__()
        self.encoder = BEVEncoder(grid_size=grid_size, scalar_dim=scalar_dim, out_dim=hidden_dim)
        self.gat = GraphAttentionLayer(in_features=hidden_dim, out_features=hidden_dim, num_heads=num_heads)
        
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, bev_grids, scalar_feats, adj_matrix):
        """
        bev_grids: (num_nodes, 2, N, N) or (batch, num_nodes, 2, N, N)
        scalar_feats: (num_nodes, 3) or (batch, num_nodes, 3)
        adj_matrix: (num_nodes, num_nodes)
        Returns:
        q_values: (num_nodes, action_dim) or (batch, num_nodes, action_dim)
        """
        is_batched = (bev_grids.dim() == 5)
        if not is_batched:
            bev_grids = bev_grids.unsqueeze(0)
            scalar_feats = scalar_feats.unsqueeze(0)

        batch_size, num_nodes, c, h, w = bev_grids.shape
        # Flatten batch and nodes for encoder
        flat_bevs = bev_grids.view(batch_size * num_nodes, c, h, w)
        flat_scalars = scalar_feats.view(batch_size * num_nodes, -1)

        node_embeds = self.encoder(flat_bevs, flat_scalars)
        node_embeds = node_embeds.view(batch_size, num_nodes, -1)

        coop_embeds = self.gat(node_embeds, adj_matrix)  # (batch, num_nodes, hidden_dim)
        q_values = self.q_head(coop_embeds)              # (batch, num_nodes, action_dim)

        if not is_batched:
            q_values = q_values.squeeze(0)
        return q_values
