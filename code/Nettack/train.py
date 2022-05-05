from models import GCN
import numpy as np
import torch as th
from utils import *
from nettack import Nettack
from args import args_parser
import os
from matplotlib import pyplot as plt

args = args_parser()

fix_seed(args.seed)
if args.save not in os.listdir():
    os.mkdir(args.save)

# Load dataset
adj, feats, lbls = load_npz(f"data/{args.dataset}.npz")

# Find largest connected components
adj = adj + adj.transpose()
adj[adj > 1] = 1
lcc = largest_connected_components(adj)

adj = adj[lcc][:, lcc]
assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"
assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

feats = feats[lcc].astype(np.float32)
lbls = lbls[lcc]

n_nodes, in_feats = feats.shape
out_feats = lbls.max()+1
degrees = adj.sum(0).A1
lbls_onehot = np.eye(out_feats)[lbls]

train_index, val_index, test_index = train_val_test_split(np.arange(n_nodes),
                                                          train_size=0.1,
                                                          val_size=0.1,
                                                          test_size=0.8,
                                                          stratify=lbls_onehot)

# Convert to tensor
adj_tensor = tuple_to_tensor(sparse_to_tuple(preprocess_adj(adj, -0.5)))
feats_tensor = tuple_to_tensor(sparse_to_tuple(feats))
lbls_tensor = th.LongTensor(lbls)

# Train surrogate model
net = GCN(in_feats, args.n_hid, out_feats, adj_tensor, with_relu=False, dropout=args.dropout)
net.retrain(feats_tensor, lbls_tensor, train_index, val_index, args)

# Nettack
target_node = args.target_node
assert target_node in test_index

atk = Nettack(adj, feats, lbls, net.w1.detach().numpy(), net.w2.detach().numpy(), target_node)

direct_attack = args.direct_attack
perturb_features = args.perturb_features
perturb_structure = args.perturb_structure
n_influencers = 1 if direct_attack else 5
n_perturbations = int(degrees[target_node])  # How many perturbations to perform. Default: Degree of the node

atk.attack_surrogate(n_perturbations, perturb_structure, perturb_features, direct_attack, n_influencers)


def evaluate(adj_eval, feats_eval):
    # Evaluate nettack
    gcn = GCN(in_feats, args.n_hid, out_feats, adj_eval)
    classification_margins = []
    class_distrs = []

    for it in range(args.retrain_iters):
        gcn.retrain(feats_eval, lbls_tensor, train_index, val_index, args, verbose=False)
        gcn.eval()
        probs = th.softmax(gcn(feats_eval)[target_node], dim=-1).detach().numpy()
        class_distrs.append(probs)
        best_second_class = (probs - 1000*lbls_onehot[target_node]).argmax()
        margin = probs[lbls[target_node]] - probs[best_second_class]
        classification_margins.append(margin)
        print(f"Train iters {it+1}/{args.retrain_iters} class_margin= {margin:.5f}")

    return np.array(class_distrs)


def visualize_results():
    def make_xlabel(ix, correct):
        if ix == correct:
            return "Class {}\n(correct)".format(ix)
        return "Class {}".format(ix)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    center_ixs_clean = []
    for ix, block in enumerate(class_distrs_clean.T):
        x_ixs = np.arange(len(block)) + ix*(len(block)+2)
        center_ixs_clean.append(np.mean(x_ixs))
        color = '#555555'
        if ix == atk.target_label:
            color = 'darkgreen'
        plt.bar(x_ixs, block, color=color)

    ax = plt.gca()
    plt.ylim((-.05, 1.05))
    plt.ylabel("Predicted probability")
    ax.set_xticks(center_ixs_clean)
    ax.set_xticklabels([make_xlabel(k, atk.target_label) for k in range(out_feats)])
    ax.set_title("Predicted class probabilities for node {} on clean data\n({} re-trainings)".format(target_node, args.retrain_iters))

    plt.subplot(1, 2, 2)
    center_ixs_retrain = []
    for ix, block in enumerate(class_distrs_perturbed.T):
        x_ixs = np.arange(len(block)) + ix*(len(block)+2)
        center_ixs_retrain.append(np.mean(x_ixs))
        color = '#555555'
        if ix == atk.target_label:
            color = 'darkgreen'
        plt.bar(x_ixs, block, color=color)

    ax = plt.gca()
    plt.ylim((-.05, 1.05))
    ax.set_xticks(center_ixs_retrain)
    ax.set_xticklabels([make_xlabel(k, atk.target_label) for k in range(out_feats)])
    ax.set_title("Predicted class probabilities for node {} after {} perturbations\n({} re-trainings)".format(target_node, n_perturbations, args.retrain_iters))

    plt.tight_layout()
    plt.savefig(f"{args.save}/{args.dataset}.png")
    plt.show()


# Train GCN without perturbations
class_distrs_clean = evaluate(adj_tensor, feats_tensor)

# Retrain GCN with perturbations
adj_tensor_perturbed = tuple_to_tensor(sparse_to_tuple(atk.adj_norm))
feats_tensor_perturbed = tuple_to_tensor(sparse_to_tuple(atk.feats_modified))
class_distrs_perturbed = evaluate(adj_tensor_perturbed, feats_tensor_perturbed)


# Visualize results
visualize_results()
