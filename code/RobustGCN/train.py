import time
from args import args_parser
from utils import *
from models import RGCN
import torch as th
import torch.nn.functional as F

args = args_parser()

# Set random seed
fix_seed(args.seed)

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data(args.dataset)

features = preprocess_features(features)

in_feats = features[2][1]
out_feats = y_train.shape[1]

# Convert to torch sparse tensor
features = tuple_to_tensor(features)
adj_list = [
    tuple_to_tensor(preprocess_adj(adj, -0.5)),
    tuple_to_tensor(preprocess_adj(adj, -1.0))
]

labels = th.from_numpy(labels)
y_train = th.from_numpy(y_train)
y_val = th.from_numpy(y_val)
y_test = th.from_numpy(y_test)
train_mask = th.from_numpy(train_mask)
val_mask = th.from_numpy(val_mask)
test_mask = th.from_numpy(test_mask)

# Create model and optimizer
model = RGCN(in_feats, out_feats, args.n_hid, adj_list, args.dropout, args.param_var)
optimizer = th.optim.Adam(model.parameters(), lr=args.lr)


def masked_cross_entropy(preds, labels, mask):
    loss = F.cross_entropy(preds, labels, reduction='none')
    mask = mask.type(th.float32)
    mask /= th.mean(mask)
    loss *= mask
    return th.mean(loss)


def masked_accuracy(preds, labels, mask):
    correct_prediction = (preds.argmax(1) == labels.argmax(1))
    accuracy_all = correct_prediction.type(th.float32)
    mask = mask.type(th.float32)
    mask /= th.mean(mask)
    accuracy_all *= mask
    return th.mean(accuracy_all)


cost_val = []

for epoch in range(args.n_epochs):
    t = time.time()
    model.train()

    preds = model(features)
    optimizer.zero_grad()

    loss = masked_cross_entropy(preds, y_train, train_mask)
    loss += args.param_l2 * th.norm(model.layers[0].weight)
    mean = model.layers[0].mean
    var = model.layers[0].var
    KL_divergence = 0.5 * th.mean(th.square(mean) + var - th.log(1e-8 + var) - 1, 1)
    KL_divergence = th.sum(KL_divergence)
    loss += args.param_kl * KL_divergence

    loss.backward()
    optimizer.step()

    train_acc = masked_accuracy(preds, y_train, train_mask)
    cost = masked_cross_entropy(preds, y_val, val_mask)
    cost_val.append(cost.item())

    # Print results
    print(f"Epoch: {epoch+1:04d} train_loss= {loss:.5f} train_acc= {train_acc:.5f} time= {time.time() - t:.5f}")
    if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping+1):-1]):
        print("Early stopping...")
        break

# Testing
model.eval()
preds = model(features, training=False)
cost = masked_cross_entropy(preds, y_test, test_mask)
acc = masked_accuracy(preds, y_test, test_mask)
print(f"Test: cost= {cost:.5f} accuracy= {acc:.5f}")
