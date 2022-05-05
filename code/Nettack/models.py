import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.metrics import f1_score
import time


class GCN(nn.Module):
    def __init__(self, in_feats, n_hid, out_feats, adj, with_relu=True, dropout=0.):
        super(GCN, self).__init__()
        self.adj = adj
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu if with_relu else None

        self.w1 = nn.Parameter(th.Tensor(in_feats, n_hid))
        self.w2 = nn.Parameter(th.Tensor(n_hid, out_feats))
        if with_relu:
            self.b1 = nn.Parameter(th.Tensor(n_hid))
            self.b2 = nn.Parameter(th.Tensor(out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.w1)
        init.xavier_uniform_(self.w2)
        if self.act:
            init.zeros_(self.b1)
            init.zeros_(self.b2)

    def forward(self, x, training=True):
        if training:
            x = sparse_dropout(x, self.dropout_rate)
        x = th.mm(x, self.w1)
        h1 = th.mm(self.adj, x)
        if self.act:
            h1 = self.act(h1 + self.b1)

        h1 = self.dropout(h1)
        h1 = th.mm(h1, self.w2)
        outputs = th.mm(self.adj, h1)
        if self.act:
            outputs = outputs + self.b2

        return outputs

    def retrain(self, feats_tensor, lbls_tensor, train_index, val_index, args, verbose=True):
        optimizer = th.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_performance = 0
        patience = args.early_stopping
        for epoch in range(args.n_epochs):
            t = time.time()
            self.train()
            preds = self(feats_tensor)
            optimizer.zero_grad()
            loss = F.cross_entropy(preds[train_index], lbls_tensor[train_index])
            acc = (preds[train_index].argmax(1) == lbls_tensor[train_index]).sum().item() / len(train_index)
            if verbose:
                print(f"Epoch: {epoch+1:04d} train_loss= {loss:.5f} train_acc= {acc:.5f} time= {time.time() - t:.5f}")
            loss.backward()
            optimizer.step()

            # Validation
            self.eval()
            preds = self(feats_tensor)
            preds = preds.argmax(1)
            f1_micro = f1_score(lbls_tensor[val_index], preds[val_index], average='micro')
            f1_macro = f1_score(lbls_tensor[val_index], preds[val_index], average='macro')
            perf_sum = f1_micro + f1_macro
            if perf_sum > best_performance:
                best_performance = perf_sum
                patience = args.early_stopping
                th.save(self.state_dict(), f"{args.save}/{args.dataset}.pkl")
            else:
                patience -= 1
            if epoch > args.early_stopping and patience <= 0:
                break

        self.load_state_dict(th.load(f"{args.save}/{args.dataset}.pkl"))


def sparse_dropout(x, dropout=0.):
    """Dropout for sparse tensors."""
    dropout_mask = th.floor(th.rand(x._values().size())+(1-dropout)).type(th.bool)
    i = x._indices()[:, dropout_mask]
    v = x._values()[dropout_mask]
    pre_out = th.sparse.FloatTensor(i, v, x.shape)
    return pre_out * (1. / (1-dropout))
