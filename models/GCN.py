import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()

        # Define a single GCN layer followed by ReLU
        self.conv = GCNConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)

        # Print GCNConv attributes to debug
        print(f'GCNConv parameters: {self.conv.__dict__}')

        # Initialize weights to 3
        self._initialize_weights()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)  # Apply ReLU activation
        return x

    def _initialize_weights(self):
        # Initialize all weights and biases to 3
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, GCNConv)):
                if hasattr(module, 'weight'):
                    nn.init.constant_(module.weight, 3)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 3)

# Example usage:
# model = GCN(in_channels=16, hidden_channels=32, out_channels=32, num_layers=2)
# out = model(data.x, data.edge_index)
