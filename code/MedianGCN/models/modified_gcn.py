import torch as th
import torch.nn as nn
import dgl.function as fn
from torch.nn import init


class ModifiedConv(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, reduce=None):
        super(ModifiedConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._activation = activation
        self._reduce = reduce
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(th.Tensor(out_feats))
        if th.cuda.is_available():
            self.weight.cuda()
            self.bias.cuda()
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, graph, feats):
        h = feats
        h = th.mm(h, self.weight)
        graph.ndata['h'] = h
        graph.update_all(fn.copy_src('h', 'm'), self._reduce)
        h = graph.ndata.pop('h')
        h = h + self.bias
        if self._activation:
            h = self._activation(h)
        return h

    def extra_repr(self):
        summary = 'in={_in_feats}, out={_out_feats}, normalization=both'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


class ModifiedGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, reduce):
        super(ModifiedGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ModifiedConv(in_feats, n_hidden, activation, reduce))
        for i in range(n_layers - 1):
            self.layers.append(ModifiedConv(n_hidden, n_hidden, activation, reduce))
        self.layers.append(ModifiedConv(n_hidden, n_classes, reduce=reduce))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(g, h)
        return h
