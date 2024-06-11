import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, x))
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class FinalLayer(nn.Module):
    def __init__(self, nfeat, nclass):
        super(FinalLayer, self).__init__()
        self.gc = GraphConvolution(nfeat, nclass)

    def forward(self, x, adj):
        x = self.gc(x, adj)
        return F.log_softmax(x, dim=1)
