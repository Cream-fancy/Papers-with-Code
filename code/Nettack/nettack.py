import numpy as np
import scipy.sparse as sp
from utils import *
from numba import jit as njit


class Nettack():
    def __init__(self, adj, feats, labels, w1, w2, target_node, verbose=True):
        # Adjacency matrix
        self.adj_orig = adj.copy().tolil()
        self.adj_modified = adj.copy().tolil()
        self.adj_norm = preprocess_adj(adj).tolil()

        # Node attributes
        self.feats_orig = feats.copy().tolil()
        self.feats_modified = feats.copy().tolil()

        # Node labels
        self.labels = labels.copy()
        self.target_label = labels[target_node]

        self.n_nodes = adj.shape[0]
        self.n_classes = np.max(self.labels)+1
        self.target_node = target_node

        # GCN weight matrices
        self.W = np.matmul(w1, w2)

        # features co-occurrence
        self.cooc_matrix = feats.T.dot(feats).tolil()
        self.cooc_constraint = None

        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []

        self.verbose = verbose

    def compute_target_logits(self):
        """
        Compute the target logits of the surrogate model.
        """

        return (self.adj_norm @ self.adj_norm @ self.feats_modified @ self.W)[self.target_node]

    def strongest_wrong_class(self, target_logits):
        """
        Determine the incorrect class with largest logits.
        """

        target_label_onehot = np.eye(self.n_classes)[self.target_label]
        return (target_logits - 1000 * target_label_onehot).argmax()

    def compute_cooccurrence_constraint(self, influencer_nodes):
        """
        Co-occurrence constraint as described in the paper.

        Parameters
        ----------
        influencer_nodes: np.array
            Nodes whose features are considered for change

        Returns
        -------
        np.array (len(influencer_nodes), n_feats), dtype bool
            True in entry n,d indicates that we are allowed to add feature d to the features of node n.

        """

        words_graph = self.cooc_matrix.copy()   # shape (D, D) like A_sq, 2 hop Neighbors feat-node-feat
        n_feats = self.feats_modified.shape[1]
        words_graph.setdiag(0)
        words_mask = (words_graph > 0)
        word_degrees = np.sum(words_mask, axis=0).A1
        inv_word_degrees = np.reciprocal(word_degrees.astype(float) + 1e-8)   # random walk

        # P(i|S_u) Addition of feature i not in S_u as unnoticeable. C.f. equation 11.
        scores_matrix = sp.lil_matrix((self.n_nodes, n_feats))
        for infl in influencer_nodes:
            common_words = words_mask.multiply(self.feats_modified[infl])   # mask of feature i and infl's feature j, E_ij in formula
            inv_degs_j = inv_word_degrees[common_words.nonzero()[1]]
            feat_inds_i = common_words.nonzero()[0]
            scores = np.array([inv_degs_j[feat_inds_i == i].sum() for i in range(n_feats)])   # (D, )
            scores_matrix[infl] = scores

        sum_d = np.zeros([self.n_nodes])
        for v in range(self.n_nodes):
            feat_inds = self.feats_modified[v, :].nonzero()[1]
            sum_d[v] = np.sum(inv_word_degrees[feat_inds])

        self.cooc_constraint = sp.csr_matrix(scores_matrix - 0.5 * sum_d[:, None] > 0)

    def gradient_wrt_x(self, label):
        """
        Compute the gradient of the logit belonging to the class of the input label with respect to the input features.

        Parameters
        ----------
        label: int
            Class whose logits are of interest

        Returns
        -------
        np.array (N, D) matrix containing the gradients.

        """

        return self.adj_norm.dot(self.adj_norm)[self.target_node].T.dot(self.W[:, label].reshape(1, -1))

    def compute_feature_scores(self):
        """
        Compute feature scores for all possible feature changes.
        """

        if self.cooc_constraint is None:
            self.compute_cooccurrence_constraint(self.influencer_nodes)
        logits = self.compute_target_logits()
        best_wrong_class = self.strongest_wrong_class(logits)
        gradient = self.gradient_wrt_x(self.target_label) - self.gradient_wrt_x(best_wrong_class)
        surrogate_loss = logits[self.target_label] - logits[best_wrong_class]

        gradients_flipped = sp.lil_matrix(gradient * -1)        # Gamma_ui (N, D)
        # flip the sign of existing feats' grads, make grads < 0 get high scores.
        gradients_flipped[self.feats_modified.nonzero()] *= -1

        X_influencers = sp.lil_matrix(self.feats_modified.shape)
        X_influencers[self.influencer_nodes] = self.feats_modified[self.influencer_nodes]
        # Features attack constraint, existing features or co-occurring features
        gradients_flipped = gradients_flipped.multiply((self.cooc_constraint + X_influencers) > 0)

        nz_inds = np.array(gradients_flipped.nonzero()).T             # (num_nonzero, 2)
        ranking = np.argsort(gradients_flipped[tuple(nz_inds.T)]).A1
        sorted_inds = nz_inds[ranking]
        grads = gradients_flipped[tuple(nz_inds[ranking].T)]

        scores = surrogate_loss - grads
        return sorted_inds[::-1], scores.A1[::-1]

    def compute_struct_scores(self, A_hat_uv, XW):
        """
        Compute structure scores, cf. Eq. 15 in the paper.
        """

        logits = A_hat_uv.dot(XW)
        label_onehot = np.eye(XW.shape[1])[self.target_label]
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:, self.target_label]
        struct_scores = logits_for_correct_class - best_wrong_class_logits
        return struct_scores

    def incremental_compute_A_hat_uv(self, potential_edges):
        """
        Compute [A_hat_square]_uv from inserting/deleting (u, v) the input edges, respectively.
        """

        degs = self.adj_modified.sum(0).A1 + 1
        edges = np.array(self.adj_modified.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = self.adj_norm @ self.adj_norm
        values_before = A_hat_sq[self.target_node].toarray()[0]
        twohop_edges = np.array(A_hat_sq.nonzero()).T

        # (N, ) ind of each node first occurrences, use to compute each node neighbors
        node_indices = np.unique(edges[:, 0], return_index=True)[1]

        inds, vals = incremental_compute_A_hat_uv(edges, twohop_edges, edges_set, node_indices, values_before, degs,
                                                  potential_edges, self.target_node)
        inds_arr = np.array(inds)
        A_hat_uv = sp.coo_matrix((vals, (inds_arr[:, 0], inds_arr[:, 1])), shape=[len(potential_edges), self.n_nodes])

        return A_hat_uv

    def get_attacker_nodes(self, n=5, add_additional_nodes=False):
        """
        Determine the influencer nodes.
        """

        assert n < self.n_nodes-1, "number of influencers cannot be >= number of nodes in the graph!"

        adj_no_self_loops = self.adj_modified.copy()
        adj_no_self_loops.setdiag(0)
        neighbors = adj_no_self_loops[self.target_node].nonzero()[1]
        assert self.target_node not in neighbors

        XW = self.feats_modified @ self.W

        # Target node neighbors (Neighbor(u), 2)
        potential_edges = np.column_stack((np.tile(self.target_node, len(neighbors)), neighbors)).astype("int32")

        # Compute the result if we removed the edge from u to each of the neighbors.
        A_hat_uv = self.incremental_compute_A_hat_uv(potential_edges)

        # Compute the struct scores for all neighbors
        struct_scores = self.compute_struct_scores(A_hat_uv, XW)

        if len(neighbors) >= n:  # do we have enough neighbors for the number of desired influencers?
            influencer_nodes = neighbors[np.argsort(struct_scores)[:n]]
            if add_additional_nodes:
                return influencer_nodes, np.array([])
            return influencer_nodes
        else:
            influencer_nodes = neighbors
            if add_additional_nodes:  # Add additional influencers by connecting them to u first.
                # Compute the set of possible additional influencers, ``Complement(Neighbor(u) Cap u)``
                poss_add_neighbors = np.setdiff1d(np.setdiff1d(np.arange(self.n_nodes), neighbors), self.target_node)
                possible_edges = np.column_stack((np.tile(self.target_node, len(poss_add_neighbors)), poss_add_neighbors))

                # Compute the struct_scores for all possible additional influencers, and choose the k-best.
                A_hat_uv_additional = self.incremental_compute_A_hat_uv(possible_edges)
                additional_struct_scores = self.compute_struct_scores(A_hat_uv_additional, XW)
                additional_influencers = poss_add_neighbors[np.argsort(additional_struct_scores)[-(n-len(neighbors))::]]

                return influencer_nodes, additional_influencers
            else:
                return influencer_nodes

    def attack_surrogate(self, n_perturbations, perturb_structure=True, perturb_features=True,
                         direct=True, n_influencers=0, delta_cutoff=0.004):
        """
        Perform an attack on the surrogate model.

        Parameters
        ----------
        n_perturbations: int
            The number of perturbations (structure or feature) to perform.

        perturb_structure: bool, default: True
            Indicates whether the structure can be changed.

        perturb_features: bool, default: True
            Indicates whether the features can be changed.

        direct: bool, default: True
            indicates whether to directly modify edges/features of the node attacked or only those of influencers.

        n_influencers: int, default: 0
            Number of influencing nodes -- will be ignored if direct is True

        delta_cutoff: float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95.

        Returns
        -------
        None.

        """

        assert not (direct == False and n_influencers == 0), "indirect mode requires at least one influencer node"
        assert n_perturbations > 0, "need at least one perturbation"
        assert perturb_features or perturb_structure, "either perturb_features or perturb_structure must be true"

        logits = self.compute_target_logits()
        best_wrong_class = self.strongest_wrong_class(logits)
        surrogate_losses = [logits[self.target_label] - logits[best_wrong_class]]

        if self.verbose:
            print("##### Starting attack #####")
            if perturb_structure and perturb_features:
                print("##### Attack node with ID {} using structure and feature perturbations #####".format(self.target_node))
            elif perturb_features:
                print("##### Attack only using feature perturbations #####")
            elif perturb_structure:
                print("##### Attack only using structure perturbations #####")
            if direct:
                print("##### Attacking the node directly #####")
            else:
                print("##### Attacking the node indirectly via {} influencer nodes #####".format(n_influencers))
            print("##### Performing {} perturbations #####".format(n_perturbations))

        if perturb_structure:
            # Setup starting values of the likelihood ratio test.
            d_min = 2
            degs_start = self.adj_orig.sum(0).A1
            S_d_start = np.sum(np.log(degs_start[degs_start >= d_min]))
            n_start = np.sum(degs_start >= d_min)
            alpha_start = compute_alpha(n_start, S_d_start, d_min)
            ll_start = compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)

            degs_current = self.adj_modified.sum(0).A1
            S_d_current = np.sum(np.log(degs_current[degs_current >= d_min]))
            n_current = np.sum(degs_current >= d_min)

        # Generate potential edges
        if len(self.influencer_nodes) == 0:
            if not direct:
                # Choose influencer nodes
                infls, add_infls = self.get_attacker_nodes(n_influencers, add_additional_nodes=True)
                self.influencer_nodes = np.concatenate((infls, add_infls)).astype("int")
                # Potential edges are all edges from any attacker to any other node, except the respective
                # attacker itself or the node being attacked.
                self.potential_edges = np.row_stack([
                    np.column_stack((
                        np.tile(infl, self.n_nodes - 2),
                        np.setdiff1d(np.arange(self.n_nodes), np.array([self.target_node, infl]))
                    )) for infl in self.influencer_nodes
                ])
                if self.verbose:
                    print("Influencer nodes: {}".format(self.influencer_nodes))
            else:
                # direct attack
                influencers = [self.target_node]
                self.potential_edges = np.column_stack((
                    np.tile(self.target_node, self.n_nodes - 1),
                    np.setdiff1d(np.arange(self.n_nodes), self.target_node)
                ))
                self.influencer_nodes = np.array(influencers)
        self.potential_edges = self.potential_edges.astype("int32")
        for _ in range(n_perturbations):
            if self.verbose:
                print("##### ...{}/{} perturbations ... #####".format(_+1, n_perturbations))
            if perturb_structure:
                # Do not consider edges that, if removed, result in singleton edges in the graph.
                singleton_filter = filter_singletons(self.potential_edges, self.adj_modified)
                filtered_edges = self.potential_edges[singleton_filter]

                # Update the values for the power law likelihood ratio test.
                deltas = -2 * self.adj_modified[tuple(filtered_edges.T)].toarray()[0] + 1
                d_edges_old = degs_current[filtered_edges]                    # (F, 2), num of filtered edges
                d_edges_new = degs_current[filtered_edges] + deltas[:, None]  # (F, 2)
                S_d_new, n_new = update_S_d(S_d_current, n_current, d_edges_old, d_edges_new, d_min)   # (F, )
                alphas_new = compute_alpha(n_new, S_d_new, d_min)
                ll_new = compute_log_likelihood(n_new, alphas_new, S_d_new, d_min)
                alphas_combined = compute_alpha(n_new + n_start, S_d_new + S_d_start, d_min)
                ll_combined = compute_log_likelihood(n_new + n_start, alphas_combined, S_d_new + S_d_start, d_min)
                ratios = -2 * ll_combined + 2 * (ll_new + ll_start)   # C.f. equation 8, 9.

                # Do not consider edges that, if added/removed, would lead to a violation of the
                # likelihood ration Chi-square cutoff value. C.f. equation 10.
                powerlaw_filter = filter_chisquare(ratios, delta_cutoff)
                filtered_edges_final = filtered_edges[powerlaw_filter]

                # Compute new entries in A_hat_square_uv
                A_hat_uv_new = self.incremental_compute_A_hat_uv(filtered_edges_final)
                # Compute the struct scores for each potential edge
                struct_scores = self.compute_struct_scores(A_hat_uv_new, self.feats_modified @ self.W)
                best_edge_idx = struct_scores.argmin()
                best_edge_score = struct_scores.min()
                best_edge = filtered_edges_final[best_edge_idx]

            if perturb_features:
                # Compute the feature scores for each potential feature perturbation
                feature_inds, feature_scores = self.compute_feature_scores()
                best_feature_idx = feature_inds[0]
                best_feature_score = feature_scores[0]

            if perturb_structure and perturb_features:
                # decide whether to choose an edge or feature to change
                if best_edge_score < best_feature_score:
                    if self.verbose:
                        print("Edge perturbation: {}".format(best_edge))
                    change_structure = True
                else:
                    if self.verbose:
                        print("Feature perturbation: {}".format(best_feature_idx))
                    change_structure = False
            elif perturb_structure:
                change_structure = True
            elif perturb_features:
                change_structure = False

            if change_structure:
                # perform edge perturbation
                self.adj_modified[tuple(best_edge)] = self.adj_modified[tuple(best_edge[::-1])] = 1 - self.adj_modified[tuple(best_edge)]
                self.adj_norm = preprocess_adj(self.adj_modified)
                self.structure_perturbations.append(tuple(best_edge))
                self.feature_perturbations.append(())
                surrogate_losses.append(best_edge_score)

                # Update likelihood ratio test values
                S_d_current = S_d_new[powerlaw_filter][best_edge_idx]
                n_current = n_new[powerlaw_filter][best_edge_idx]
                degs_current[best_edge] += deltas[powerlaw_filter][best_edge_idx]

            else:
                self.feats_modified[tuple(best_feature_idx)] = 1 - self.feats_modified[tuple(best_feature_idx)]
                self.feature_perturbations.append(tuple(best_feature_idx))
                self.structure_perturbations.append(())
                surrogate_losses.append(best_feature_score)


@njit(nopython=True)
def incremental_compute_A_hat_uv(edges, twohop_edges, edges_set, node_indices, values_before, degs, potential_edges, u):
    """
    Compute the new values [A_hat_square]_u for every potential edge, where u is the target node. C.f. Theorem 5.1
    equation 17.

    Parameters
    ----------
    edges: shape (?, 2)
        The indices of the nodes connected by the edges in the input graph.

    twohop_edges: shape (?, 2)
        The indices of nodes that are in the twohop neighborhood of each other, including self-loops.

    edges_set: set((e0, e1))
        The set of edges in the input graph.

    node_indices: shape (N, )
        For each node, this gives the first index of edges associated to this node in the edge array.
        This will be used to quickly look up the neighbors of a node, since numba does not allow nested lists.

    values_before: shape (N, )
        The values in [A_hat]^2_uv to be updated.

    degs: shape (N, )
        The degree of the nodes in the input graph.

    potential_edges: shape (P, 2)
        The potential edges to be evaluated. For each of these potential edges, this function will compute the values
        in [A_hat]^2_uv that would result after inserting/removing this edge.

    u: int
        The target node

    Returns
    -------
    return_indices: List of tuples
        The indices in the (P, N) matrix of updated values that have changed.
    return_values:
    """

    n_nodes = degs.shape[0]

    # 2-hop-Neighbor(u)
    nbs_u = edges[edges[:, 0] == u, 1]
    nbs_u_twohop = twohop_edges[twohop_edges[:, 0] == u, 1]
    nbs_u_set = set(nbs_u)

    return_indices = []
    return_values = []

    for idx in range(len(potential_edges)):
        edge = potential_edges[idx]
        edge_set = set(edge)
        degs_after = degs.copy()
        delta = -2 * ((edge[0], edge[1]) in edges_set) + 1  # insert/remove edge
        degs_after[edge] += delta

        # Compute A_um, A_un, A'_um, A'_un
        a_um_before = edge[0] in nbs_u_set
        a_un_before = edge[1] in nbs_u_set
        a_um_after = (delta > 0 if u == edge[1] else a_um_before)
        a_un_after = (delta > 0 if u == edge[0] else a_un_before)

        # Affected nodes include 2-hop-Neighbor(u), Neighbor(m), Neighbor(n), m, n
        # NOTE: Why need to update 2-hop-Neighbor(u)
        nbs_m = edges[edges[:, 0] == edge[0], 1]
        nbs_n = edges[edges[:, 0] == edge[1], 1]
        affected_nodes = set(np.concatenate((nbs_u_twohop, nbs_m, nbs_n)))
        affected_nodes = affected_nodes.union(edge_set)

        for v in affected_nodes:
            # A_uv, A_tilde_uv
            a_uv_before = v in nbs_u_set
            a_uv_before_sl = a_uv_before or v == u
            a_uv_after = (delta > 0 if (v in edge_set and u in edge_set and u != v) else a_uv_before)
            a_uv_after_sl = a_uv_after or v == u

            # Find Neighbor(v)
            from_idx = node_indices[v]
            to_idx = node_indices[v + 1] if v < n_nodes - 1 else len(edges)
            v_nbs = edges[from_idx:to_idx, 1]
            v_nbs_set = set(v_nbs)

            a_vm_before = edge[0] in v_nbs_set
            a_vn_before = edge[1] in v_nbs_set
            a_vn_after = (delta > 0 if v == edge[0] else a_vn_before)
            a_vm_after = (delta > 0 if v == edge[1] else a_vm_before)

            mult_term = 1 / np.sqrt(degs_after[u] * degs_after[v])
            sum_term1 = np.sqrt(degs[u] * degs[v]) * values_before[v] - a_uv_before_sl / degs[u] - a_uv_before / degs[v]
            sum_term2 = a_uv_after / degs_after[v] + a_uv_after_sl / degs_after[u]
            sum_term3 = -((a_um_before and a_vm_before) / degs[edge[0]]) + (a_um_after and a_vm_after) / degs_after[edge[0]]
            sum_term4 = -((a_un_before and a_vn_before) / degs[edge[1]]) + (a_un_after and a_vn_after) / degs_after[edge[1]]
            new_val = mult_term * (sum_term1 + sum_term2 + sum_term3 + sum_term4)

            return_indices.append((idx, v))
            return_values.append(new_val)

    return return_indices, return_values


def compute_alpha(n, S_d, d_min):
    """
    Approximate the alpha of a power law distribution. C.f. equation 6.
    """

    return n / (S_d - n * np.log(d_min - 0.5)) + 1


def compute_log_likelihood(n, alpha, S_d, d_min):
    """
    Compute log likelihood of the powerlaw fit. C.f. equation 7.
    """

    return n * np.log(alpha) + n * alpha * np.log(d_min) - (alpha + 1) * S_d


def filter_singletons(potential_edges, adj):
    """
    Filter edges that, if removed, would turn one or more nodes into singleton nodes.

    Returns
    -------
    np.array, shape (P, ), dtype bool:
        A binary vector of length len(potential_edges), False values indicate that the edge at
        the index  generates singleton edges, and should thus be avoided.
    """

    degs = np.squeeze(np.array(np.sum(adj, 0)))                                       # (N, )
    existing_edge_mask = np.squeeze(np.array(adj.tocsr()[tuple(potential_edges.T)]))  # (P, ), num of potential edges
    edge_degrees = degs[np.array(potential_edges)]                                    # (P, 2)
    edge_degrees = edge_degrees - 2 * existing_edge_mask[:, None] + 1                 # add/remove edge
    zeros_sum = (edge_degrees == 0).sum(1)
    return zeros_sum == 0


def update_S_d(S_d_old, n_old, d_old, d_new, d_min):
    """
    Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
    a single edge.

    Parameters
    ----------
    S_d_old: float
        Sum of log degrees in the old distribution that are larger than or equal to d_min.

    n_old: int
        Number of entries in the old distribution that are larger than or equal to d_min.

    d_old: np.array, shape (N, 2), dtype int
        The old degree sequence.

    d_new: np.array, shape (N, 2), dtype int
        The new degree sequence

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    S_d_new: float, shape (N, )
        the updated sum of log degrees in the distribution that are larger than or equal to d_min.
    n_new: int, shape (N, )
        the updated number of entries in the old distribution that are larger than or equal to d_min.
    """

    old_mask = d_old >= d_min
    new_mask = d_new >= d_min

    d_old_in_range = d_old * old_mask
    d_new_in_range = d_new * new_mask

    # When ``d = 0``, max(d, 1) ensure log(d) equal to zero instead of -inf.
    S_d_new = S_d_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
    n_new = n_old - np.sum(old_mask, 1) + np.sum(new_mask, 1)

    return S_d_new, n_new


def filter_chisquare(ll_ratios, cutoff):
    """
    Chi-square independence test.
    """

    return ll_ratios < cutoff
