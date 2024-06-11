import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.relus.append(nn.ReLU(inplace=False))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.relus.append(nn.ReLU(inplace=False))

        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv, relu in zip(self.convs[:-1], self.relus):
            x = conv(x, edge_index)
            x = relu(x)
        x = self.convs[-1](x, edge_index)  # No ReLU on the last layer
        return x

# Example usage:
# model = GCN(in_channels=16, hidden_channels=32, out_channels=2, num_layers=4)
# out = model(data.x, data.edge_index)
