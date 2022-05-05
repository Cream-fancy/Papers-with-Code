import torch as th

trimmed_ratio = 0.2


def gcn_reduce(nodes):
    h = nodes.data['out_norm'] * th.sum(nodes.mailbox['m'], 1) * nodes.data['in_norm']
    return {'h': h}


def median_reduce(nodes):
    h = th.median(nodes.mailbox['m'], dim=1).values
    return {'h': h}


def trimmed_reduce(nodes):
    h = nodes.mailbox['m']
    n_neighbor = h.shape[1]
    n_trimmed = int(trimmed_ratio * n_neighbor)
    h = th.sort(h, 1).values
    h = h[:, n_trimmed:n_neighbor-n_trimmed]
    h = nodes.data['out_norm'] * th.sum(h, 1) * nodes.data['in_norm']
    return {'h': h}
