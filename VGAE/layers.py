import torch
import torch.nn as nn
from utils import glorot_init


class DropoutSparse(nn.Module):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """

    def __init__(self, dropout, feats_nonzero):
        super(DropoutSparse, self).__init__()
        self.keep_prob = 1-dropout
        self.noise_shape = [feats_nonzero]

    def forward(self, x):
        random_tensor = self.keep_prob
        random_tensor += torch.rand(self.noise_shape)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        pre_out = torch.sparse.FloatTensor(torch.LongTensor(i), torch.FloatTensor(v), torch.Size(x.shape))
        return pre_out * (1. / self.keep_prob)


class GraphConv(nn.Module):
    """Graph convolution layer for dense or sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, dropout=0., activation=lambda x: x, feats_nonzero=0, is_sparse=False, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.dropout = nn.Dropout(dropout)  # Dropout placed in `forward()` would cause parameter not to update.
        self.dropout_sparse = DropoutSparse(dropout, feats_nonzero)
        self.activation = activation
        self.is_sparse = is_sparse

    def forward(self, inputs):
        x = inputs
        if self.is_sparse:
            x = self.dropout_sparse(x)
        else:
            x = self.dropout(x)
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, dropout=0., act=torch.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z):
        z = self.dropout(z)
        A_pred = self.act(torch.matmul(z, z.t()))
        return A_pred
