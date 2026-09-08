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
        h: (num_nodes, in_features)
        adj: (num_nodes, num_nodes) adjacency matrix
        """
        num_nodes = h.size(0)
        # Linear projection: (num_nodes, num_heads, head_dim)
        Wh = self.W(h).view(num_nodes, self.num_heads, self.head_dim)

        # Attention logits: e_ij = a_src * Wh_i + a_dst * Wh_j
        attn_src = (Wh * self.a_src).sum(dim=-1)  # (num_nodes, num_heads)
        attn_dst = (Wh * self.a_dst).sum(dim=-1)  # (num_nodes, num_heads)

        # Broadcast addition for all pairs (i, j): (num_nodes, num_nodes, num_heads)
        attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)
        attn = self.leaky_relu(attn)

        # Mask non-connected edges
        mask = (adj == 0).unsqueeze(-1)  # (num_nodes, num_nodes, 1)
        attn = attn.masked_fill(mask, -1e9)

        # Softmax over neighbors (dim=1)
        alpha = F.softmax(attn, dim=1)
        alpha = self.dropout(alpha)  # (num_nodes, num_nodes, num_heads)

        # Weighted sum: (num_nodes, num_heads, head_dim)
        # alpha: (num_nodes, num_nodes, num_heads) -> permute to (num_heads, num_nodes, num_nodes)
        # Wh: (num_nodes, num_heads, head_dim) -> permute to (num_heads, num_nodes, head_dim)
        alpha_p = alpha.permute(2, 0, 1)
        Wh_p = Wh.permute(1, 0, 2)
        h_prime = torch.bmm(alpha_p, Wh_p)  # (num_heads, num_nodes, head_dim)
        h_prime = h_prime.permute(1, 0, 2).contiguous().view(num_nodes, self.out_features)

        return F.elu(h_prime)


class CoLightNetwork(nn.Module):
    """
    Complete CoLight MARL Network:
    BEVEncoder -> GAT Message Passing -> Action-Value (Q-Network) Head.
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
        bev_grids: (num_nodes, 2, N, N)
        scalar_feats: (num_nodes, 3)
        adj_matrix: (num_nodes, num_nodes)
        Returns:
        q_values: (num_nodes, action_dim)
        """
        # 1. Local state embedding
        node_embeddings = self.encoder(bev_grids, scalar_feats)  # (num_nodes, hidden_dim)

        # 2. Inter-intersection cooperative message passing
        coop_embeddings = self.gat(node_embeddings, adj_matrix)  # (num_nodes, hidden_dim)

        # 3. Action-value prediction
        q_values = self.q_head(coop_embeddings)  # (num_nodes, action_dim)
        return q_values
