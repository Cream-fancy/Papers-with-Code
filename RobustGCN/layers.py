import torch as th
import torch.nn as nn
import torch.nn.functional as F
from inits import glorot, zeros


class GGCL_F(nn.Module):
    """GGCL: the input is feature"""

    def __init__(self, in_feats, out_feats, adj_list, dropout=0., param_var=1, is_sparse=False, bias=False, featureless=False, elu=F.elu, relu=F.relu):
        super(GGCL_F, self).__init__()
        self.adj_list = adj_list
        self.dropout_rate = dropout
        self.is_sparse = is_sparse
        self.featureless = featureless
        self.out_feats = out_feats
        self.param_var = param_var

        self.weight = glorot(in_feats, out_feats)
        if bias:
            self.bias = zeros(out_feats)

        self.act_mean = elu
        self.act_var = relu

    def forward(self, inputs, training=True):
        x = inputs
        # Manually shut `sparse_dropout` and `F.dropout` off in `model.eval`
        if training:
            if self.is_sparse:
                x = sparse_dropout(x, self.dropout_rate)
            else:
                x = F.dropout(x, self.dropout_rate)
        if not self.featureless:
            hidden = th.mm(x, self.weight)
        else:
            hidden = self.weight
        dim = int(self.out_feats / 2)
        mean = self.act_mean(hidden[:, 0:dim])
        variance = self.act_var(hidden[:, dim:dim*2])
        self.mean = mean
        self.var = variance

        node_weight = th.exp(-variance * self.param_var)
        mean_out = th.mm(self.adj_list[0], mean * node_weight)
        var_out = th.mm(self.adj_list[1], variance * node_weight * node_weight)

        outputs = th.concat([mean_out, var_out], dim=1)
        return outputs


class GGCL_D(nn.Module):
    """GGCL: the input is distribution (mean, variance)"""

    def __init__(self, in_feats, out_feats, adj_list, dropout=0., param_var=1, is_sparse=False, bias=False, featureless=False, elu=F.elu, relu=F.relu):
        super(GGCL_D, self).__init__()
        self.adj_list = adj_list
        self.dropout = nn.Dropout(dropout)
        self.dim = int(in_feats / 2)
        self.param_var = param_var

        self.weight_mean = glorot(self.dim, out_feats)
        self.weight_var = glorot(self.dim, out_feats)
        if bias:
            self.bias = zeros(out_feats)

        self.act_mean = elu
        self.act_var = relu

    def forward(self, inputs, training=True):
        x = inputs
        mean = x[:, 0:self.dim]
        variance = x[:, self.dim:self.dim*2]
        mean = self.dropout(mean)
        variance = self.dropout(variance)
        mean = self.act_mean(th.mm(mean, self.weight_mean))
        variance = self.act_var(th.mm(variance, self.weight_var))

        node_weight = th.exp(-variance * self.param_var)
        mean_out = th.mm(self.adj_list[0], mean * node_weight)
        var_out = th.mm(self.adj_list[1], variance * node_weight * node_weight)

        sampled_v = th.randn(var_out.shape)
        mean_out = mean_out + (th.sqrt(var_out + 1e-8) * sampled_v)
        outputs = mean_out
        return outputs


def sparse_dropout(x, dropout=0.):
    """Dropout for sparse tensors."""
    dropout_mask = th.floor(th.rand(x._values().size())+(1-dropout)).type(th.bool)
    i = x._indices()[:, dropout_mask]
    v = x._values()[dropout_mask]
    pre_out = th.sparse.FloatTensor(i, v, x.shape)
    return pre_out * (1. / (1-dropout))
