import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout, alpha, concat=True, device='cuda'):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha
        self.concat = concat
        self.device = device

        # 各ヘッドに対して重み行列と注意機構を定義
        self.W = nn.ParameterList([nn.Parameter(torch.empty(size=(in_features, out_features)).to(device)) for _ in range(num_heads)])
        for w in self.W:
            nn.init.xavier_uniform_(w.data, gain=1.414)

        self.a = nn.ParameterList([nn.Parameter(torch.empty(size=(2 * out_features, 1)).to(device)) for _ in range(num_heads)])
        for a in self.a:
            nn.init.xavier_uniform_(a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, h, adj):
        Wh_all_heads = []
        for head in range(self.num_heads):
            Wh = torch.mm(h, self.W[head])  # 各ヘッドで重み行列を適用
            e = self._prepare_attentional_mechanism_input(Wh, head)

            zero_vec = -9e15 * torch.ones_like(e).to(self.device)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = self.dropout(attention)
            h_prime = torch.matmul(attention, Wh)
            Wh_all_heads.append(h_prime)

        if self.concat:
            # 全てのヘッドの出力を結合
            return F.elu(torch.cat(Wh_all_heads, dim=1))
        else:
            # 平均を取る場合
            return torch.mean(torch.stack(Wh_all_heads), dim=0)

    def _prepare_attentional_mechanism_input(self, Wh, head):
        Wh1 = torch.matmul(Wh, self.a[head][:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[head][self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class AdjacencyGenerator(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, device, dropout=0.6, alpha=0.2):
        super(AdjacencyGenerator, self).__init__()
        self.num_layers = num_layers
        self.device = device

        # Define Multi-Head Attention Layers
        self.attention_layers = nn.ModuleList([
            MultiHeadGraphAttentionLayer(d_model, d_model // num_heads, num_heads, dropout, alpha, device=device) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout).to(device)
        self.weight_vector = nn.Linear(d_model, 1).to(device)
        self.leaky_relu = nn.LeakyReLU(0.2).to(device)  # Add LeakyReLU instance

    def forward(self, edge_index, x):
        num_nodes = x.size(0)
        adj_logits_all_layers = []

        for layer in range(self.num_layers):
            x_i = x[edge_index[0]]
            x_j = x[edge_index[1]]

            # Apply Multi-Head Attention
            adj_logits = self.attention_layers[layer](x_j, edge_index)
            adj_logits_all_layers.append(adj_logits)

            # Update features for the next layer
            x = F.elu(adj_logits)
            x = self.dropout(x)

        # Combine logits from all layers
        adj_logits = torch.mean(torch.stack(adj_logits_all_layers), dim=0)

        # Apply the final transformation
        adj_logits = self.leaky_relu(adj_logits)
        adj_logits = self.weight_vector(adj_logits).squeeze(1)

        return adj_logits

    def generate_new_neighbors(self, edge_index, x):
        adj_logits = self.forward(edge_index, x)
        adj_probs = torch.sigmoid(adj_logits / 20).to(self.device)  # Reduce to (num_neighbors + 1)
        new_edges = torch.bernoulli(adj_probs).to(self.device)  # Sample new neighbors

        return adj_logits, new_edges

    def extract_new_edges(self, edge_index, new_edges):
        return edge_index[:, new_edges == 1]

# 使用例
d_model = 128
num_heads = 8
num_layers = 3
dropout = 0.6
alpha = 0.2
num_nodes = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn(num_nodes, d_model).to(device)  # ノードの特徴量
edge_index = torch.randint(0, num_nodes, (2, num_nodes)).to(device)  # エッジのインデックス

adj_generator = AdjacencyGenerator(d_model, num_heads, num_layers, device, dropout, alpha).to(device)
adj_logits, new_edges = adj_generator.generate_new_neighbors(edge_index, x)
print(f"adj_logits: {adj_logits}")
print(f"new_edges: {new_edges}")
