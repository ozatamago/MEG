import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.6):
        super(GCN, self).__init__()

        # Define a single GCN layer followed by ReLU
        self.conv = GCNConv(in_channels, out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.conv(x, edge_index)
        x = self.leaky_relu(x)  # Apply ReLU activation
        return x

# Example usage:
# model = GCN(in_channels=16, hidden_channels=32, out_channels=32, num_layers=2)
# out = model(data.x, data.edge_index)
