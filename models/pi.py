import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

class AdjacencyGenerator(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, device, dropout=0.1):
        super(AdjacencyGenerator, self).__init__()
        self.num_layers = num_layers
        self.device = device
        
        # Define layers
        self.query_layers = nn.ModuleList([
            nn.Linear(d_model, d_model).to(device) for _ in range(num_layers)
        ])
        self.key_layers = nn.ModuleList([
            nn.Linear(d_model, d_model).to(device) for _ in range(num_layers)
        ])
        self.value_layers = nn.ModuleList([
            nn.Linear(d_model, d_model).to(device) for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model).to(device) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout).to(device)

        self.weight_layer = nn.Linear(d_model, 4 * d_model).to(device)
        self.weight_layer2 = nn.Linear(4 * d_model, 2 * d_model).to(device)
        self.weight_layer3 = nn.Linear(2 * d_model, d_model).to(device)
        self.weight_vector = nn.Linear(d_model, 1).to(device)

        self.final_norm = nn.LayerNorm(d_model).to(device)

    def get_attention(self, edge_index, query, key, num_nodes):
        alpha = (query * key).sum(-1)
        alpha = softmax(alpha, edge_index[1], num_nodes=num_nodes)
        return alpha

    def forward(self, edge_index, x):
        num_nodes = x.size(0)
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]

        for i in range(self.num_layers):
            query = self.query_layers[i](x_i)
            key = self.key_layers[i](x_j)
            value = self.value_layers[i](x_j)

            attn_output_weights = self.get_attention(edge_index, query, key, num_nodes)
            attn_output_weights = attn_output_weights.unsqueeze(1)  # Make it 2D (query_length, 1)
            attn_output = attn_output_weights * value
            attn_output = self.dropout(attn_output)  # Apply Dropout
            attn_output = F.relu(attn_output)  # Apply ReLU activation

            query = x_j + attn_output  # Skip connection with original query
            query = self.norm_layers[i](query)  # Norm

        adj_logits = attn_output.squeeze(0)  # (1, d_model) -> (d_model)
        
        adj_logits = F.linear(adj_logits, self.weight_layer.weight.clone(), self.weight_layer.bias)
        adj_logits = self.dropout(adj_logits)  # Apply Dropout
        adj_logits = F.linear(adj_logits, self.weight_layer2.weight.clone(), self.weight_layer2.bias)
        adj_logits = self.dropout(adj_logits)  # Apply Dropout
        adj_logits = F.linear(adj_logits, self.weight_layer3.weight.clone(), self.weight_layer3.bias)
        
        adj_logits = adj_logits + query
        adj_logits = self.final_norm(adj_logits)

        adj_logits = F.linear(adj_logits, self.weight_vector.weight.clone(), self.weight_vector.bias).squeeze(1)

        return adj_logits

    def generate_new_neighbors(self, edge_index, x):
        adj_logits = self.forward(edge_index, x)
        adj_probs = torch.sigmoid(adj_logits / 50).to(self.device)  # Reduce to (num_neighbors + 1)
        new_edges = torch.bernoulli(adj_probs).to(self.device)  # Sample new neighbors

        return adj_logits, new_edges

    def extract_new_edges(self, edge_index, new_edges):
        return edge_index[:, new_edges == 1]

# # Example usage
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long).to(device)
# x = torch.tensor([[-1.0], [1.0], [2.0]], dtype=torch.float).to(device)

# model = AdjacencyGenerator(d_model=1, num_heads=1, num_layers=2, device=device, dropout=0.2)
# adj_logits, new_edges = model.generate_new_neighbors(edge_index, x)

# print("Adjacency logits:", adj_logits)
# print("New edges:", new_edges)

# filtered_edges = model.extract_new_edges(edge_index, new_edges)
# print("Filtered edges:", filtered_edges)
