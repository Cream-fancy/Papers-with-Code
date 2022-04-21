import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='MedianGCN')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer'], help="dataset name")
    parser.add_argument('--model',  type=str, default='gcn', choices=['gcn', 'median', 'trimmed'], help="model name")
    parser.add_argument('--influence-attack', action='store_true', default=False, help="attack type (default is directed attack)")
    parser.add_argument('--n-epochs',  type=int, default=200, help="number of training epochs")
    parser.add_argument('--lr',  type=float, default=0.01, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden units")
    parser.add_argument("--n-layers", type=int, default=1, help="number of hidden layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true', default=False, help="graph self-loop (default=False)")
    return parser.parse_args()
