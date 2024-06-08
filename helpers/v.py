import torch
import torch.nn as nn

class VNetwork(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, num_samples, dropout=0.1):
        super(VNetwork, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear1 = nn.Linear(d_model, 1)  # Output single value
        self.linear2 = nn.Linear(num_samples, 1)  # Output single value

    def forward(self, x):
        for layer in self.encoder.layers:
            sa_output, _ = layer.self_attn(x, x, x, need_weights=False)
            sa_output = sa_output.clone()  # Add this line to avoid in-place modification
            sa_output = layer.dropout1(sa_output)
            sa_output = layer.norm1(x + sa_output)

            ff_output = layer.linear1(sa_output)
            ff_output = layer.dropout(layer.activation(ff_output))
            ff_output = layer.linear2(ff_output)
            x = layer.norm2(sa_output + layer.dropout2(ff_output))

        output = nn.functional.linear(x, self.linear1.weight.clone(), self.linear1.bias).squeeze(2)
        output = nn.functional.linear(output, self.linear2.weight.clone(), self.linear2.bias)
        return output
