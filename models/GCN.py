import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()

        # Define a single GCN layer followed by ReLU
        self.conv = GCNConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)

        # Initialize weights to 3
        self._initialize_weights()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)  # Apply ReLU activation
        return x

    def _initialize_weights(self):
        # Initialize weights and biases to 3 for GCNConv
        nn.init.constant_(self.conv.lin.weight, 3)  # self.conv.lin is a torch.nn.Linear layer inside GCNConv
        if self.conv.lin.bias is not None:
            nn.init.constant_(self.conv.lin.bias, 3)
        nn.init.constant_(self.conv.root.weight, 3)  # self.conv.root is also a torch.nn.Linear layer inside GCNConv
        if self.conv.root.bias is not None:
            nn.init.constant_(self.conv.root.bias, 3)

# Example usage:
# model = GCN(in_channels=16, hidden_channels=32, out_channels=32, num_layers=2)
# out = model(data.x, data.edge_index)
