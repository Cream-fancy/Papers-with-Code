import numpy as np
import torch as th
import dgl
import scipy.sparse as sp
import random


def normalize(norm, graph):
    out_degs = graph.out_degrees().float()
    in_degs = graph.in_degrees().float()
    if norm == 'left':
        out_norm = 1. / out_degs
    elif norm == 'right':
        in_norm = 1. / in_degs
    else:
        out_norm = th.pow(out_degs, -0.5)
        in_norm = th.pow(in_degs, -0.5)
    if out_norm is not None:
        out_norm[th.isinf(out_norm)] = 0
        out_norm = out_norm.unsqueeze(1)
    if in_norm is not None:
        in_norm[th.isinf(in_norm)] = 0
        in_norm = in_norm.unsqueeze(1)
    return out_norm, in_norm


def adv_attack(g, test_mask, args):
    labels = g.ndata['label']
    n_nodes = g.num_nodes()

    adv_edges = np.load(f'adversarial_edges/targeted/{args.dataset}_Nettack{"_In" if args.influence_attack else ""}.npy', allow_pickle=True).item()
    targets = list(adv_edges.keys())
    targets = np.array([i for i in targets if i in test_mask])
    target = targets[np.random.randint(0, len(targets))]
    target_label = labels[target].item()
    num_budgets = int((g.in_degrees(target) + 1) / 2) + 1

    print(f"Attack target node {target} with classes {target_label}. \nThe degree of target node {target} is {g.in_degrees(target)}, thus we use {num_budgets} as attack budgets")

    flip_edges = adv_edges[target][:num_budgets]
    S = th.sparse_coo_tensor(indices=th.tensor(flip_edges).T, values=th.ones(len(flip_edges)), size=(n_nodes, n_nodes))
    A = g.adj()
    _A = (th.ones(n_nodes, n_nodes) - A).to_sparse()

    atk_adj = A + (_A - A) * S
    atk_adj_indices = atk_adj._indices()
    sp_atk_adj = sp.coo_matrix((th.ones(atk_adj_indices[0].shape), (atk_adj_indices[0], atk_adj_indices[1])), atk_adj.shape)
    adv_g = dgl.from_scipy(sp_atk_adj)
    return adv_g, target


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
