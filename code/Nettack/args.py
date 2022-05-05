import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Nettack')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'polblogs'], help="dataset name")
    parser.add_argument('--save', type=str, default='temp', help="dir for saving model or perturbed data")
    parser.add_argument('--n_epochs',  type=int, default=200, help="number of training epochs")
    parser.add_argument('--retrain_iters',  type=int, default=5, help="number of retraining iterations")
    parser.add_argument("--n_hid", type=int, default=16, help="number of hidden units")
    parser.add_argument('--lr',  type=float, default=0.01, help="learning rate")
    parser.add_argument('--weight_decay',  type=float, default=5e-4, help="l2 weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--early_stopping", type=int, default=30,  help="tolerance for early stopping (# of epochs)")

    # nettack
    parser.add_argument('--target_node', type=int, default=1)
    parser.add_argument('--direct_attack', type=int, default=1)
    parser.add_argument('--perturb_structure', type=int, default=1)
    parser.add_argument('--perturb_features', type=int, default=1)
    return parser.parse_args()
