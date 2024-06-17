import torch
import torch.nn as nn

class FinalLayer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FinalLayer, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
        # Initialize weights to 3
        self._initialize_weights()

    def forward(self, x):
        return nn.functional.linear(x, self.linear.weight.clone(), self.linear.bias)

    def _initialize_weights(self):
        # Initialize all weights and biases to 3
        nn.init.constant_(self.linear.weight, 3)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 3)

# Example usage:
# model = FinalLayer(input_dim=64, num_classes=10)
# out = model(torch.randn(1, 64))
