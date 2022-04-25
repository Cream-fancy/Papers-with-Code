import time
import torch as th
import torch.nn.functional as F
from args import args_parser
from models import GCN
from utils import *
from metrics import *

args = args_parser()

fix_seed(args.seed)

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)

features = preprocess_features(features)
adj = preprocess_adj(adj)

in_feats = features[2][1]
out_feats = y_train.shape[1]
n_feats_nonzero = features[1].shape[0]

# Convert to torch sparse tensor
features = tuple_to_tensor(features)
adj = tuple_to_tensor(adj)

y_train = th.from_numpy(y_train)
y_val = th.from_numpy(y_val)
y_test = th.from_numpy(y_test)
train_mask = th.from_numpy(train_mask)
val_mask = th.from_numpy(val_mask)
test_mask = th.from_numpy(test_mask)

# Create model and optimizer
model = GCN(in_feats, args.n_hid, out_feats, adj, act=F.relu, dropout_rate=args.dropout)
optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

cost_val = []
for epoch in range(args.n_epochs):
    t = time.time()

    # Training step
    model.train()
    logits = model(features)
    optimizer.zero_grad()
    loss = masked_cross_entropy(logits, y_train, train_mask)
    # loss += args.weight_decay * model.layers[0].l2_loss()
    loss.backward()
    optimizer.step()

    acc = masked_accuracy(logits, y_train, train_mask)
    print(f"Epoch: {epoch+1:04d} train_loss= {loss:.5f} train_acc= {acc:.5f} time= {time.time() - t:.5f}")

    # Validation
    model.eval()
    with th.no_grad():
        logits = model(features)
        cost = masked_cross_entropy(logits, y_val, val_mask).item()
        cost_val.append(cost)

    if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping+1):-1]):
        break

model.eval()
cost = masked_cross_entropy(logits, y_test, test_mask).item()
acc = masked_accuracy(logits, y_test, test_mask).item()
print(f"Test cost= {cost:.5f} acc= {acc:.5f}")
