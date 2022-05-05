from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components
import torch as th
import random


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file."""
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)

        adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                             loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            feats = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                   loader['attr_indptr']), shape=loader['attr_shape'])

        else:
            feats = None
            # N = loader['adj_shape'][0]
            # D = np.max(loader['labels'])+1
            # shp = (N, D)
            # feats = sp.coo_matrix((np.ones(N*D), (np.repeat(np.arange(N), D), np.zeros(N*D))), shape=shp).tocsr()

        labels = loader['labels']

    return adj, feats, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph."""
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return nodes_to_keep


def train_val_test_split(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):
    """
    Split the arrays or matrices into random train, validation and test subsets."""
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for arr in arrays:
        result.append(arr[idx_train])
        result.append(arr[idx_val])
        result.append(arr[idx_test])
    return result


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj, alpha, norm='both'):
    """Normalize adjacency matrix."""
    adj_norm = adj = sp.coo_matrix(adj)
    if norm == 'left' or norm == 'both':
        rowsum = np.array(adj.sum(1))   # out degrees
        d_inv_sqrt = np.power(rowsum, alpha).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj_norm = d_mat_inv_sqrt.dot(adj_norm)
    if norm == 'right' or norm == 'both':
        colsum = np.array(adj.sum(0))   # in degrees
        d_inv_sqrt = np.power(colsum, alpha).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj_norm = adj_norm.dot(d_mat_inv_sqrt)
    return adj_norm.tocoo()


def preprocess_adj(adj, alpha=-0.5, norm='both'):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = adj + adj.transpose()
    adj = adj + sp.eye(adj.shape[0])
    adj[adj > 1] = 1
    adj_normalized = normalize_adj(adj, alpha, norm)
    adj_normalized.eliminate_zeros()
    return adj_normalized


def tuple_to_tensor(input):
    """Convert tuple representation to torch sparse tensor."""
    return th.sparse.FloatTensor(th.LongTensor(input[0].T), th.FloatTensor(input[1]), th.Size(input[2]))


def fix_seed(seed=0):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
