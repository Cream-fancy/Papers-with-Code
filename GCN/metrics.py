import torch as th
import torch.nn.functional as F


def masked_cross_entropy(preds, labels, mask):
    """cross-entropy loss with masking."""
    loss = F.cross_entropy(preds, labels, reduction='none')
    mask = mask.type(th.float32)
    mask /= th.mean(mask)
    loss *= mask
    return th.mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = (preds.argmax(1) == labels.argmax(1))
    accuracy_all = correct_prediction.type(th.float32)
    mask = mask.type(th.float32)
    mask /= th.mean(mask)
    accuracy_all *= mask
    return th.mean(accuracy_all)
