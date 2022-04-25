import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConv


class GCN(nn.Module):
    def __init__(self, in_feats, n_hid, out_feats, adj, act=F.relu, dropout_rate=0.):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hid, adj, act=act, dropout=True))
        self.layers.append(GraphConv(n_hid, out_feats, adj))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
            if layer.dropout:
                h = self.dropout(h)
        return h
