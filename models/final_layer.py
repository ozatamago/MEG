import torch
import torch.nn as nn

class FinalLayer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FinalLayer, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return nn.functional.linear(x, self.linear.weight.clone(), self.linear.bias)
