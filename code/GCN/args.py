import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'], help="dataset name")
    parser.add_argument('--n-epochs',  type=int, default=200, help="number of training epochs")
    parser.add_argument('--lr',  type=float, default=0.01, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--n-hid", type=int, default=16, help="number of hidden units")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument('--seed',  type=int, default=123, help="random seed")
    parser.add_argument("--early-stopping", type=int, default=20,  help="tolerance for early stopping (# of epochs)")
    return parser.parse_args()
