from re import A
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import time

from utils import *
from args import args_parser
from models import VGAE, GAE

args = args_parser()

# Load data
adj, features = load_data(args.dataset)

if args.use_feats == 0:
    features = sp.identity(features.shape[0])   # featureless

# Store original adjacency matrix (witorch.ut diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

adj_label = tuple_to_tensor(adj_label)
adj_norm = tuple_to_tensor(adj_norm)
features = tuple_to_tensor(features)

# Create model and optimizer
if args.model == 'vgae':
    model = VGAE(num_features, args.n_hid1, args.n_hid2, adj_norm, features_nonzero, args.dropout)
else:
    model = GAE(num_features, args.n_hid1, args.n_hid2, adj_norm, features_nonzero, args.dropout)
optimizer = Adam(model.parameters(), lr=args.lr)


def get_scores(edges_pos, edges_neg, adj_rec):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


# Balance the positive and negative edge weights in BCE loss
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()   # n_pos * w_pos = n_neg * w_neg, where w_neg = 1
weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
weight_tensor[weight_mask] = pos_weight

# Normalize BCE loss, suppose L_neg approx L_pos
# norm * (n_pos * w_pos * L_pos + n_neg * 1 * L_neg) = (n_neg + n_pos) * L
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)   # (n_pos + n_neg) / (n_neg * 2)

# Train model
for epoch in range(args.n_epochs):
    t = time.time()

    A_pred = model(features)
    optimizer.zero_grad()
    loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor, reduction='mean')

    # Latent loss
    if args.model == 'vgae':
        """Compute KL_div[q(Z|X,A), p(Z)], choose Gaussian prior p(z) ~ N(0,1)

        NOTE: The official code implements the KL divergence sign as the opposite of the formula.
              Instead, I use the original KL divergence.
              Minimize loss here, not maximize objective like paper.
        """
        kl_divergence = (0.5 / num_nodes) * (torch.square(model.z_mean) + torch.square(torch.exp(model.z_log_std)) - 2 * model.z_log_std - 1).sum(1).mean()
        loss += kl_divergence

    loss.backward()
    optimizer.step()

    train_acc = get_acc(A_pred, adj_label)

    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap),
          "time=", "{:.5f}".format(time.time() - t))


test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print('Test ROC score: ' + str(test_roc))
print('Test AP score: ' + str(test_ap))
