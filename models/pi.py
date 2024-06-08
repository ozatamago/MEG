import torch
import torch.nn as nn
from models.transformer import TransformerEncoder

class AdjacencyGenerator(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, hidden_size, device, dropout=0.1):
        super(AdjacencyGenerator, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout).to(device)
        self.device = device

        self.weight_layer = nn.Linear(d_model, 2*d_model).to(device)
        self.weight_layer2 = nn.Linear(2*d_model, 2*d_model).to(device)
        self.weight_vector = nn.Linear(2*d_model, 1).to(device)

    def forward(self, node_features, neighbor_features):
        input_features = torch.cat([node_features, neighbor_features], dim=0).unsqueeze(0)  # (1, num_neighbors + 1, d_model)
        adj_logits = self.encoder(input_features.clone())  # (num_neighbors + 1, 1, d_model)        

        # for layer in self.encoder.encoder.layers:
        #     # print(f"layer: {layer}")
        #     sa_output, _ = layer.self_attn(input_features, input_features, input_features, need_weights=False)
        #     sa_output = layer.dropout1(sa_output)
        #     sa_output = layer.norm1(input_features + sa_output).clone()

        #     # Clone weights and biases for feed-forward network
        #     ff_output = torch.nn.functional.linear(sa_output, layer.linear1.weight.clone(), layer.linear1.bias.clone())
        #     ff_output = layer.activation(ff_output)
        #     ff_output = layer.dropout(ff_output).clone()      
        #     ff_output = torch.nn.functional.linear(ff_output, layer.linear2.weight.clone(), layer.linear2.bias.clone())
        #     ff_output = layer.dropout2(ff_output).clone()
        #     input_features = layer.norm2(sa_output + ff_output).clone()

        adj_logits = nn.functional.linear(input_features, self.weight_layer.weight.clone(), self.weight_layer.bias)
        adj_logits = nn.functional.linear(adj_logits, self.weight_layer2.weight.clone(), self.weight_layer2.bias)
        adj_logits = nn.functional.linear(adj_logits, self.weight_vector.weight.clone(), self.weight_vector.bias).squeeze(1)
        adj_probs = torch.sigmoid(adj_logits / 3).to(self.device)  # Reduce to (num_neighbors + 1)


        return adj_probs, adj_logits


    def generate_new_neighbors(self, node_features, neighbor_features):
        adj_probs, adj_logits = self.forward(node_features, neighbor_features)
        new_neighbors = torch.bernoulli(adj_probs).to(self.device)  # Sample new neighbors

        return adj_logits, new_neighbors