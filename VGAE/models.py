import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConv, InnerProductDecoder


class VGAE(nn.Module):
    def __init__(self, n_feats, n_hid1, n_hid2, adj, feats_nonzero, dropout=0.):
        super(VGAE, self).__init__()
        self.graph_conv1 = GraphConv(n_feats, n_hid1, adj, dropout, activation=F.relu, feats_nonzero=feats_nonzero, is_sparse=True)
        self.graph_conv_mean = GraphConv(n_hid1, n_hid2, adj, dropout)
        self.graph_conv_logstddev = GraphConv(n_hid1, n_hid2, adj, dropout)
        self.n_hid2 = n_hid2
        self.decode = InnerProductDecoder(dropout)

    def encode(self, x):
        hidden = self.graph_conv1(x)
        embeddings = self.graph_conv_mean(hidden)
        self.z_mean = embeddings
        self.z_log_std = self.graph_conv_logstddev(hidden)

        # sample z from N(\mu,\sigma) using reparameterization trick
        gaussian_noise = torch.randn(x.size(0), self.n_hid2)                  # sample eps from N(0,1)
        sampled_z = self.z_mean + torch.exp(self.z_log_std) * gaussian_noise  # z = \mu + \sigma * eps
        return sampled_z

    def forward(self, x):
        z = self.encode(x)
        A_pred = self.decode(z)
        return A_pred


class GAE(nn.Module):
    def __init__(self, n_feats, n_hid1, n_hid2, adj, feats_nonzero, dropout=0.):
        super(GAE, self).__init__()
        self.graph_conv1 = GraphConv(n_feats, n_hid1, adj, dropout, activation=F.relu, feats_nonzero=feats_nonzero, is_sparse=True)
        self.graph_conv2 = GraphConv(n_hid1, n_hid2, adj, dropout)
        self.decode = InnerProductDecoder(dropout)

    def encode(self, x):
        hidden = self.graph_conv1(x)
        embeddings = self.graph_conv2(hidden)
        return embeddings

    def forward(self, x):
        z = self.encode(x)
        A_pred = self.decode(z)
        return A_pred
