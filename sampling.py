import numpy as np
from torch.nn.functional import cross_entropy
import torch


def none(pred_classes, true_classes):
    return pred_classes, true_classes


def undersample(pred_classes, true_classes, random_state=0):
    rng = np.random.RandomState(random_state)
    ct = true_classes
    cp = pred_classes
    ind = torch.arange(len(ct))
    pos = ind[(ct.data.cpu() > 0)].long().cuda()
    neg = ind[(ct.data.cpu() == 0)].long().cuda()
    ct_pos = ct[pos]
    cp_pos = cp[pos]
    ct_neg = ct[neg]
    cp_neg = cp[neg]
    nb = len(ct_pos)
    inds = torch.from_numpy(rng.randint(0, len(ct_neg), nb))
    inds = inds.long().cuda()
    ct_neg = ct_neg[inds]
    cp_neg = cp_neg[inds]
    true = torch.cat((ct_pos, ct_neg), dim=0)
    pred = torch.cat((cp_pos, cp_neg), dim=0)
    return pred, true


def oversample(pred_classes, true_classes, negative_per_positive=3, random_state=0):
    rng = np.random.RandomState(random_state)
    ct = true_classes
    cp = pred_classes
    ind = torch.arange(len(ct))
    pos = ind[(ct.data.cpu() > 0)].long().cuda()
    neg = ind[(ct.data.cpu() == 0)].long().cuda()
    ct_pos = ct[pos]
    cp_pos = cp[pos]
    ct_neg = ct[neg]
    cp_neg = cp[neg]
    nb = len(ct_neg)
    inds = torch.from_numpy(
        rng.randint(0, len(ct_pos), nb // negative_per_positive))
    inds = inds.long().cuda()
    ct_pos = ct_pos[inds]
    cp_pos = cp_pos[inds]
    true = torch.cat((ct_pos, ct_neg), dim=0)
    pred = torch.cat((cp_pos, cp_neg), dim=0)
    return pred, true


def hard_negative_mining(pred_classes, true_classes, batch_size, loss_func=cross_entropy, negative_per_positive=3):
    true_classes_orig = true_classes.view(batch_size, -1)
    pos = true_classes_orig > 0
    nb_pos = (true_classes_orig > 0).long().sum(dim=1).view(-1, 1)
    nb_neg = nb_pos * negative_per_positive
    class_loss = loss_func(
        pred_classes, true_classes, reduce=False)
    class_loss = class_loss.view(true_classes_orig.size(0), -1)
    class_loss[pos] = 0
    _, inds = class_loss.sort(1, descending=True)
    _, ranks = inds.sort(1)
    mask = (ranks < nb_neg) | pos
    mask = mask.view(-1)
    return pred_classes[mask], true_classes[mask]
