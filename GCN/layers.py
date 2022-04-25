import torch as th
import torch.nn as nn
from torch.nn import init


class GraphConv(nn.Module):
    """Graph convolution layer."""

    def __init__(self, in_feats, out_feats, adj, act=lambda x: x, dropout=False, bias=True):
        super(GraphConv, self).__init__()
        self.act = act
        self.adj = adj
        self.dropout = dropout

        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        x = th.mm(x, self.weight)
        x = th.mm(self.adj, x)
        if self.bias is not None:
            x = x + self.bias
        return self.act(x)

    def l2_loss(self):
        loss = 0
        for param in self.parameters():
            loss += param.pow(2).sum()
        return loss
