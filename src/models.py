import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, APPNP


class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class MLP(nn.Module):
    def __init__(self, i, h, o, dropout=0.5):
        super().__init__()
        self.l1, self.l2, self.p = nn.Linear(i, h), nn.Linear(h, o), dropout

    def forward(self, x, edge_index=None):
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.l1(x))
        x = F.dropout(x, self.p, self.training)
        return self.l2(x)


class GAT(nn.Module):
    def __init__(self, i, o, heads=8, hidden=8, dropout=0.6):
        super().__init__()
        self.c1 = GATConv(i, hidden, heads=heads, dropout=dropout)
        self.c2 = GATConv(hidden * heads, o, heads=1, dropout=dropout)
        self.p = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.p, self.training)
        x = F.elu(self.c1(x, edge_index))
        x = F.dropout(x, self.p, self.training)
        return self.c2(x, edge_index)


class APPNPNet(nn.Module):
    def __init__(self, i, h, o, K=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.l1, self.l2, self.p = nn.Linear(i, h), nn.Linear(h, o), dropout
        self.prop = APPNP(K, alpha)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.l1(x))
        x = F.dropout(x, self.p, self.training)
        return self.prop(self.l2(x), edge_index)
