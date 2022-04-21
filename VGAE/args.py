import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='VGAE')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'], help="dataset name")
    parser.add_argument('--model',  type=str, default='vgae', choices=['vgae', 'gae'], help="model name")
    parser.add_argument('--n-epochs',  type=int, default=200, help="number of training epochs")
    parser.add_argument('--lr',  type=float, default=0.01, help="learning rate")
    parser.add_argument("--n-hid1", type=int, default=32, help="number of hidden1 units")
    parser.add_argument("--n-hid2", type=int, default=16, help="number of hidden2 units")
    parser.add_argument("--dropout", type=float, default=0., help="dropout rate")
    parser.add_argument("--weight-decay", type=float, default=0., help="Weight for L2 loss")
    parser.add_argument("--use-feats", type=int, default=1, choices=[0, 1], help="Whether to use features (1) or not (0)")
    return parser.parse_args()
