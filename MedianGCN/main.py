import torch as th
import torch.nn.functional as F
from dgl.data import CoraGraphDataset, CiteseerGraphDataset
from arguments import args_parser
# from models.median_dgl import MedianGCN
# from models.gcn_dgl import GCN
from models.modified_gcn import ModifiedGCN
from utils import process, reduce
from argparse import ArgumentError

args = args_parser()

if args.dataset == 'cora':
    g = CoraGraphDataset()[0]
elif args.dataset == 'citeseer':
    g = CiteseerGraphDataset()[0]
else:
    raise ArgumentError(None, f"dataset '{args.dataset}' was not found.")

feats = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
n_nodes = g.num_nodes()
in_feats = feats.shape[1]
n_classes = labels.max().item()+1

# self loop
if args.self_loop:
    g.remove_self_loop()
    g.add_self_loop()

# symmetry normalization
out_norm, in_norm = process.normalize('both', g)
if out_norm is not None:
    g.ndata['out_norm'] = out_norm
if in_norm is not None:
    g.ndata['in_norm'] = in_norm

if args.model == 'gcn':
    reduce_func = reduce.gcn_reduce
elif args.model == 'median':
    reduce_func = reduce.median_reduce
elif args.model == 'trimmed':
    reduce_func = reduce.trimmed_reduce
else:
    raise ArgumentError(None, f"model '{args.model}' was not defined.")

model = ModifiedGCN(in_feats, args.n_hidden, n_classes, args.n_layers, activation=F.relu, dropout=args.dropout, reduce=reduce_func)
opt = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

process.fix_seed()

if th.cuda.is_available():
    model.cuda()
    feats = feats.cuda()
    labels = labels.cuda()
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()
    g = g.to('cuda:0')


def evaluate(graph, mask):
    model.eval()
    with th.no_grad():
        logits = model(graph, feats)
        correct = th.sum(logits[mask].argmax(1) == labels[mask])
        return correct.item() * 1.0 / len(labels[mask]), logits


def train(graph):
    model.reset_parameters()
    best_acc = 0
    for epoch in range(args.n_epochs + 1):
        model.train()
        logits = model(graph, feats)
        opt.zero_grad()
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        acc, _ = evaluate(graph, val_mask)
        print(loss, acc)
        if acc > best_acc:
            best_acc = acc
            th.save(model.state_dict(), f'{args.model}_model.pkl')
        loss.backward()
        opt.step()


if __name__ == '__main__':
    train(g)
    model.load_state_dict(th.load(f'{args.model}_model.pkl'))
    acc_before, logits_before = evaluate(g, test_mask)

    adv_g, target = process.adv_attack(g, test_mask, args)
    if th.cuda.is_available():
        adv_g = adv_g.to('cuda:0')
    out_norm, in_norm = process.normalize('both', adv_g)
    if out_norm is not None:
        adv_g.ndata['out_norm'] = out_norm
    if in_norm is not None:
        adv_g.ndata['in_norm'] = in_norm
    tar_lb = labels[target]

    acc_after, logits_after = evaluate(adv_g, test_mask)
    print(f"before accuracy: {acc_before}, after accuracy: {acc_after}")
    print(f'before predict: {logits_before[target][tar_lb]}, after predict: {logits_after[target][tar_lb]}')
    print(f"Classification margin: {logits_before[target][tar_lb]-logits_after[target][tar_lb]}")
