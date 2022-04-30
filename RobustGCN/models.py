from re import X
import torch.nn as nn
from layers import *


class RGCN(nn.Module):
    def __init__(self, in_feats, out_feats, n_hid, adj_list, dropout=0., param_var=1):
        super(RGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GGCL_F(in_feats, n_hid, adj_list, dropout, param_var, is_sparse=True))
        self.layers.append(GGCL_D(n_hid, out_feats, adj_list, dropout, param_var))

    def forward(self, inputs, training=True):
        x = inputs
        for layer in self.layers:
            x = layer(x, training)
        outputs = x
        return outputs
