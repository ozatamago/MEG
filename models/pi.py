import torch
import torch.nn as nn
from models.transformer import TransformerEncoder

class AdjacencyGenerator(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, hidden_size, device, dropout=0.1):
        super(AdjacencyGenerator, self).__init__()
        self.num_layers = num_layers
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout).to(device) 
            for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model).to(device)
            for _ in range(num_layers)
        ])
        self.device = device

        self.weight_layer = nn.Linear(d_model, 2*d_model).to(device)
        self.weight_layer2 = nn.Linear(2*d_model, 2*d_model).to(device)
        self.weight_layer3 = nn.Linear(2*d_model, d_model).to(device)
        self.weight_vector = nn.Linear(d_model, 1).to(device)

        # Add&Norm for final logits
        self.final_norm = nn.LayerNorm(d_model).to(device)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 10)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 10)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.constant_(m.in_proj_weight, 10)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 10)
                nn.init.constant_(m.out_proj.weight, 10)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 10)

    def forward(self, node_features, neighbor_features):
        # Queryはnode_features、KeyとValueはneighbor_features
        original_query = neighbor_features.unsqueeze(1)  # (num_neighbors, 1, d_model)
        query = original_query
        key = node_features.unsqueeze(1)  # (1, num_nodes, d_model)
        value = node_features.unsqueeze(1)  # (1, num_nodes, d_model)

        for i in range(self.num_layers):
            attn_output, attn_output_weights = self.cross_attentions[i](query, key, value)
            attn_output = nn.functional.relu(attn_output)  # Apply ReLU activation
            query = original_query + attn_output  # Skip connection with original query
            query = self.norm_layers[i](query)  # Norm

        adj_logits = attn_output.squeeze(0)  # (1, d_model) -> (d_model)
        
        adj_logits = nn.functional.linear(adj_logits, self.weight_layer.weight.clone(), self.weight_layer.bias)
        # adj_logits = nn.functional.linear(adj_logits, self.weight_layer2.weight.clone(), self.weight_layer2.bias)
        adj_logits = nn.functional.linear(adj_logits, self.weight_layer3.weight.clone(), self.weight_layer3.bias)
        
        # Apply Add&Norm for final logits
        adj_logits = adj_logits + query
        adj_logits = self.final_norm(adj_logits)

        adj_logits = nn.functional.linear(adj_logits, self.weight_vector.weight.clone(), self.weight_vector.bias).squeeze(1)
        adj_probs = torch.sigmoid(adj_logits / 50).to(self.device)  # Reduce to (num_neighbors + 1)

        return adj_probs, adj_logits

    def generate_new_neighbors(self, node_features, neighbor_features):
        adj_probs, adj_logits = self.forward(node_features, neighbor_features)
        new_neighbors = torch.bernoulli(adj_probs).to(self.device)  # Sample new neighbors

        return adj_logits, new_neighbors
