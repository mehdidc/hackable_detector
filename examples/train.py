import os
import sys
import time
sys.path.append('..')
import numpy as np
from scipy.stats import norm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import smooth_l1_loss, cross_entropy, binary_cross_entropy_with_logits
from torch.optim import Adam, SGD, Adagrad
from torch.optim.lr_scheduler import StepLR, ExponentialLR


from albumentations import Compose
from albumentations import Resize
from albumentations import Normalize
from albumentations import HorizontalFlip
from albumentations import RandomCrop

from anchors import Anchors
from match import match
from torchvision.models import vgg16
from detector import DetectorBuilder
from loss import FocalLoss
from nms import non_maximal_suppression
from optim import CyclicLR

from data.base import DetectionDataset
from data.coco import get_annotations
from visualization import draw_bounding_boxes

from tensorboardX import SummaryWriter


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def prediction_layer(in_channels, out_channels):
    seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
    )
    return seq


if __name__ == '__main__':
    writer = SummaryWriter(log_dir='log')
    # hypers
    batch_size = 16
    nb_epochs = 10000
    image_size = 300 
    # dataset
    filenames, annotations = get_annotations(
        # images_folder='alcohol/images/train',
        # annotations_file='alcohol/annotations/train.json'
        images_folder='coco/train2014',
        annotations_file='coco/annotations/instances_train2014.json'
    )
    # nb_samples = 100
    # filenames = filenames[0:nb_samples]
    # annotations = annotations[0:nb_samples]
    #annotations = [[(box, 'alcohol') for box, class_id in anns] for anns in annotations]
    nb_classes = 1 + len(set(class_id for ann in annotations for box, class_id in ann))
    print('Nb classes : {}'.format(nb_classes))
    # detector model
    vgg = vgg16(pretrained=True).features
    pretrained_layers = [vgg[i] for i in range(10)]
    
    builder = DetectorBuilder()
    builder.add_layers(pretrained_layers, init=False)
    builder.add_layers([
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
    ])
    builder.add_prediction_branch(scale=0.15, aspect_ratios=[1, 2, 1/2])
    builder.add_layers([
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
    ])
    builder.add_prediction_branch(scale=0.3, aspect_ratios=[1, 2, 1/2])
    builder.add_layers([
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
    ])
    builder.add_prediction_branch(scale=0.45, aspect_ratios=[1, 2, 1/2])
    builder.add_layers([
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
    ])
    builder.add_prediction_branch(scale=0.6, aspect_ratios=[1, 2, 1/2])
    builder.add_layers([
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
    ])
    builder.add_prediction_branch(scale=0.75, aspect_ratios=[1, 2, 1/2])
    builder.add_layers([
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=2),
    ])
    builder.add_prediction_branch(scale=0.9, aspect_ratios=[1, 2, 1/2])

    detector = builder.build(
        nb_classes=nb_classes, 
        prediction_layer_func=prediction_layer
    )
    anchors = Anchors.from_detector(
        detector, 
        input_shape=(3, image_size, image_size)
    )
    detector = detector.cuda()
    transform = Compose([
        HorizontalFlip(p=0.0),
        Resize(height=image_size, width=image_size, p=1.0),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.),
    ])
    dataset = DetectionDataset(
        filenames, 
        annotations, 
        anchors,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=1,
    )
    #class_loss_func = FocalLoss(gamma=2, alpha=0.25)
    class_loss_func = cross_entropy
    #class_loss_func = nn.functional.mse_loss
    #optimizer = SGD(detector.parameters(), lr=3e-4, momentum=0.9)
    #optimizer  = Adagrad(detector.parameters(), lr=1e-3)
    optimizer = Adam(detector.parameters(), lr=1e-3)
    #scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-4, step_size=18000)
    #scheduler = StepLR(optimizer, step_size=20000, gamma=0.1)
    imbalance_strategy = 'hard_negative_mining'
    #scheduler = ExponentialLR(optimizer, gamma=1.01)
    #scheduler = StepLR(optimizer, step_size=10, gamma=1.3)
    # Load
    if os.path.exists('model.th'):
        model_dict = torch.load('model.th')
        detector =  model_dict['model']
        optimizer = model_dict['optimizer']
        nb_iter = model_dict['nb_iter']
        detector.cuda()
        detector.train()
    else:
        nb_iter = 0
    model_dict = {
        'model': detector, 
        'anchors': anchors, 
        'transform': transform,
        'optimizer': optimizer,
        'nb_iter': nb_iter
    }
    torch.save(model_dict, 'model.th')
    losses = []
    set_lr(optimizer, 1e-4)
    for epoch in range(nb_epochs):
        for batch_index, (images, encoded_boxes, classes) in enumerate(dataloader):
            t0 = time.time()
            images = images.cuda()
            encoded_boxes = encoded_boxes.cuda()
            classes = classes.cuda().long()

            pred_boxes, pred_classes = detector.predict(images)
            
            encoded_boxes = encoded_boxes.view(-1, 4)
            pred_boxes = pred_boxes.view(-1, 4)
            
            pred_classes_orig = pred_classes
            classes_orig = classes

            pred_classes = pred_classes.view(-1, detector.nb_classes)
            classes = classes.view(-1)

            pos = (classes > 0).nonzero().view(-1)
            nb_pos = len(classes[pos])

            loc_loss = smooth_l1_loss(
                pred_boxes[pos], 
                encoded_boxes[pos], 
                size_average=False) / nb_pos
            if imbalance_strategy == 'none':
                class_loss = class_loss_func(
                    pred_classes, 
                    classes, 
                    size_average=False) / nb_pos
            elif imbalance_strategy == 'undersampling':
                ct = classes
                cp = pred_classes
                ind = torch.arange(len(ct))
                pos = ind[(ct.data.cpu() > 0)].long().cuda()
                neg = ind[(ct.data.cpu() == 0)].long().cuda()
                ct_pos = ct[pos]
                cp_pos = cp[pos]
                ct_neg = ct[neg]
                cp_neg = cp[neg]
                nb = len(ct_pos)
                inds = torch.from_numpy(np.random.randint(0, len(ct_neg), nb))
                inds = inds.long().cuda()
                ct_neg = ct_neg[inds]
                cp_neg = cp_neg[inds]
                true = torch.cat((ct_pos, ct_neg), dim=0)
                pred = torch.cat((cp_pos, cp_neg), dim=0)
                class_loss = class_loss_func(pred, true, size_average=False) / nb_pos
            elif imbalance_strategy == 'oversampling':
                ct = classes
                cp = pred_classes
                ind = torch.arange(len(ct))
                pos = ind[(ct.data.cpu() > 0)].long().cuda()
                neg = ind[(ct.data.cpu() == 0)].long().cuda()
                ct_pos = ct[pos]
                cp_pos = cp[pos]
                ct_neg = ct[neg]
                cp_neg = cp[neg]
                nb = len(ct_neg)
                inds = torch.from_numpy(np.random.randint(0, len(ct_pos), nb // 3))
                inds = inds.long().cuda()
                ct_pos = ct_pos[inds]
                cp_pos = cp_pos[inds]
                true = torch.cat((ct_pos, ct_neg), dim=0)
                pred = torch.cat((cp_pos, cp_neg), dim=0)
                class_loss = class_loss_func(pred, true)
            elif imbalance_strategy == 'hard_negative_mining':
                pos = classes_orig > 0
                nb_pos = (classes_orig > 0).long().sum(dim=1).view(-1, 1)
                nb_neg = nb_pos * 3
                class_loss = class_loss_func(pred_classes, classes, reduce=False)
                class_loss = class_loss.view(classes_orig.size(0), -1)
                class_loss[pos] = 0
                _, inds = class_loss.sort(1, descending=True)
                _, ranks = inds.sort(1)
                mask = (ranks < nb_neg) | pos
                mask = mask.view(-1)
                class_loss = class_loss_func(
                    pred_classes[mask], 
                    classes[mask], 
                    size_average=False) / nb_pos.sum()
            # calculate acc on positive examples and negative ones
            ct = classes
            cp = pred_classes
            ind = torch.arange(len(ct))
            pos = ind[(ct.data.cpu() > 0)].long().cuda()
            neg = ind[(ct.data.cpu() == 0)].long().cuda()
            ct_pos = ct[pos]
            cp_pos = cp[pos]
            ct_neg = ct[neg]
            cp_neg = cp[neg]
            _, pred_class = cp_pos.max(dim=1)
            pos_acc = (pred_class == ct_pos).float().mean()
            
            _, pred_class = cp_neg.max(dim=1)
            neg_acc = (pred_class == ct_neg).float().mean()
            # train
            is_train = True
            detector.zero_grad()
            loss = loc_loss + class_loss
            loss.backward()
            """
            losses.append(loss.item())
            check_interval = 500
            if nb_iter % check_interval == 0:
                y = losses[len(losses) - check_interval:]
                x = np.linspace(0, 1, len(y))
                x = x.reshape((-1, 1))
                x = np.concatenate((x, np.ones_like(x)), axis=1)
                A, _, _, _ = np.linalg.lstsq(x, y)
                slope, _ = A
                ypred = np.dot(A, x.T)
                var_line = ((ypred - y) ** 2).sum() / (len(x) - 2)
                var_slope = (12 * var_line) / (len(x)**3 - len(x))
                prob_neg = norm.cdf(0, loc=slope, scale=np.sqrt(var_slope))
                prob_pos = 1 - prob_neg
                if prob_pos > 0.5:
                    #print('Changing learning rate')
                    #new_lr = max(get_lr(optimizer) / 10.0, 1e-5)
                    #set_lr(optimizer, new_lr)
                elif prob_neg < 0.5:
                    print('Chaning learning rate')
                    new_lr = min(get_lr(optimizer) * 10, 1e-2)
                    set_lr(optimizer, new_lr)
            """
            if is_train:
                optimizer.step()
                writer.add_scalar('data/loss', loss.item(), nb_iter)
                writer.add_scalar('data/loc', loc_loss.item(), nb_iter)
                writer.add_scalar('data/class', class_loss.item(), nb_iter)
                writer.add_scalar('data/lr', get_lr(optimizer), nb_iter)
                writer.add_scalar('data/pos_acc', pos_acc.item(), nb_iter)
                writer.add_scalar('data/neg_acc', neg_acc.item(), nb_iter)
            else:
                writer.add_scalar('data/val_loss', loss.item(), nb_iter)
                writer.add_scalar('data/val_loc', loc_loss.item(), nb_iter)
                writer.add_scalar('data/val_class', class_loss.item(), nb_iter)
                writer.add_scalar('data/lr', get_lr(optimizer), nb_iter)
                writer.add_scalar('data/val_pos_acc', pos_acc.item(), nb_iter)
                writer.add_scalar('data/val_neg_acc', neg_acc.item(), nb_iter)
            print('Epoch {:05d}/{:05d} Iter {:05d} Batch {:05d}/{:05d} Loss : {:.3f} Loc : {:.3f} '
                    'Classif : {:.3f} Pos acc : {:.4f} Neg acc : {:.4f} Time:{:.3f}s'.format(
                      epoch,
                      nb_epochs,
                      nb_iter,
                      batch_index,
                      len(dataloader), 
                      loss.item(), 
                      loc_loss.item(),
                      class_loss.item(),
                      pos_acc.item(),
                      neg_acc.item(),
                      time.time() - t0,
                    ))
            nb_iter += 1
            if nb_iter % 100 == 0:
                model_dict['nb_iter'] = nb_iter
                print('Saving...')
                torch.save(model_dict, 'model.th')


