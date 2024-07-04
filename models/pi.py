import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

class MultiheadAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, num_heads=1):
        super(MultiheadAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_heads = num_heads

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features * num_heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features * num_heads, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features * num_heads)
        Wh = Wh.view(-1, self.num_heads, self.out_features)  # Shape: (N, num_heads, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.einsum("ijk,ijl->ikl", attention, Wh)

        if self.concat:
            return F.elu(h_prime.view(-1, self.out_features * self.num_heads))
        else:
            return h_prime.mean(dim=1)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.einsum("ijk,kl->ijl", Wh, self.a[:self.out_features * self.num_heads])
        Wh2 = torch.einsum("ijk,kl->ijl", Wh, self.a[self.out_features * self.num_heads:])
        e = Wh1 + Wh2.transpose(0, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class AdjacencyGenerator(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, device, dropout=0.6):
        super(AdjacencyGenerator, self).__init__()
        self.num_layers = num_layers
        self.device = device

        self.attention_layers = nn.ModuleList([
            MultiheadAttentionLayer(d_model, d_model, dropout, alpha=0.2, concat=True, num_heads=num_heads).to(device) for _ in range(num_layers)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Linear(d_model, d_model).to(device) for _ in range(num_layers)
        ])
        self.norm_layers_a = nn.ModuleList([
            nn.LayerNorm(d_model).to(device) for _ in range(num_layers)
        ])
        self.norm_layers_f = nn.ModuleList([
            nn.LayerNorm(d_model).to(device) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout).to(device)

        self.weight_layer = nn.Linear(d_model, 3 * d_model).to(device)
        self.weight_layer5 = nn.Linear(3 * d_model, d_model).to(device)
        self.weight_vector = nn.Linear(d_model, 1).to(device)

        self.final_norm = nn.LayerNorm(d_model).to(device)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def get_attention(self, edge_index, query, key, num_nodes):
        alpha = (query * key).sum(-1)
        alpha = softmax(alpha, edge_index[1], num_nodes=num_nodes)
        return alpha

    def forward(self, edge_index, x):
        num_nodes = x.size(0)
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]

        query = x_j

        for i in range(self.num_layers):
            attention_layer = self.attention_layers[i]
            query = attention_layer(query, x_i)
            query = self.norm_layers_a[i](query)
            query = self.ff_layers[i](query)

        query = self.leaky_relu(query)
        query = self.norm_layers_f[0](query)
        query = self.dropout(query)

        adj_logits = query.squeeze(0)
        adj_logits = F.linear(adj_logits, self.weight_layer.weight.clone(), self.weight_layer.bias)
        adj_logits = self.leaky_relu(adj_logits)
        adj_logits = self.dropout(adj_logits)
        adj_logits = F.linear(adj_logits, self.weight_layer5.weight.clone(), self.weight_layer5.bias)
        adj_logits = adj_logits + query
        adj_logits = self.final_norm(adj_logits)
        adj_logits = F.linear(adj_logits, self.weight_vector.weight.clone(), self.weight_vector.bias).squeeze(1)

        return adj_logits

    def generate_new_neighbors(self, edge_index, x):
        adj_logits = self.forward(edge_index, x)
        adj_probs = torch.sigmoid(adj_logits / 20).to(self.device)
        new_edges = torch.bernoulli(adj_probs).to(self.device)
        return adj_logits, new_edges

    def extract_new_edges(self, edge_index, new_edges):
        return edge_index[:, new_edges == 1]

# テスト用コード
if __name__ == "__main__":
    d_model = 16
    num_heads = 4
    num_layers = 2
    dropout = 0.6
    num_nodes = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(num_nodes, d_model).to(device)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2)).to(device)
    adj_generator = AdjacencyGenerator(d_model, num_heads, num_layers, device, dropout)

    adj_logits, new_edges = adj_generator.generate_new_neighbors(edge_index, x)
    print("Adjacency logits:", adj_logits)
    print("New edges:", new_edges)
