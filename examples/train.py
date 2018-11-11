import sys
sys.path.append('..')  # NOQA
import os
from clize import run
from tensorboardX import SummaryWriter
from bounding_boxes import XMIN, YMIN, WIDTH, HEIGHT, decode_bounding_boxes
from visualization import draw_bounding_boxes
from data.coco import get_annotations
from data.base import DetectionDataset, load_image
from loss import FocalLoss
from detector import DetectorBuilder
from torchvision.models import vgg16
from anchors import build_anchors_matrix_from_detector
from match import match_ssd_method, match_bijective_method
from albumentations import HorizontalFlip
from albumentations import Normalize
from albumentations import Resize
from albumentations import Compose
from torch.optim import Adam
from torch.nn.functional import smooth_l1_loss, cross_entropy
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from skimage.transform import resize
import numpy as np
import time

import sampling as sampling_strategy

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


def main(*, images_folder, annotations_file):
    writer = SummaryWriter(log_dir='log')
    # hypers
    batch_size = 4
    nb_epochs = 10000
    image_size = 300
    # dataset
    filenames, annotations = get_annotations(
        images_folder=images_folder,
        annotations_file=annotations_file,
    )
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
    anchors = build_anchors_matrix_from_detector(
        detector,
        input_shape=(3, image_size, image_size)
    )
    detector = detector.cuda()
    normalize = Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225), 
        max_pixel_value=255.
    )
    transform = Compose([
        HorizontalFlip(p=0.0),
        Resize(height=image_size, width=image_size, p=1.0),
        normalize,
    ])
    dataset = DetectionDataset(
        filenames,
        annotations,
        encode_boxes=True,
        anchors=anchors,
        match_method=match_bijective_method,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    class_loss_func = FocalLoss(gamma=2)
    optimizer = Adam(detector.parameters(), lr=1e-3)
    imbalance_strategy = sampling_strategy.none 
    # Load
    if os.path.exists('model.th'):
        model_dict = torch.load('model.th')
        detector = model_dict['model']
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
    set_lr(optimizer, 1e-4)
    for epoch in range(nb_epochs):
        for batch_index, (images, encoded_boxes, classes, filenames) in enumerate(dataloader):
            t0 = time.time()
            images = images.cuda()
            encoded_boxes = encoded_boxes.cuda()
            classes = classes.cuda().long()

            pred_boxes, pred_classes = detector.predict(images)
            pred_boxes_orig = pred_boxes
            pred_classes_orig = pred_classes
            encoded_boxes_orig = encoded_boxes
            classes_orig = classes

            encoded_boxes = encoded_boxes.view(-1, 4)
            pred_boxes = pred_boxes.view(-1, 4)

            pred_classes = pred_classes.view(-1, detector.nb_classes)
            classes = classes.view(-1)
            pos = (classes > 0).nonzero().view(-1)
            nb_pos = len(classes[pos])
            loc_loss = smooth_l1_loss(
                pred_boxes[pos],
                encoded_boxes[pos],
                size_average=False) / nb_pos
            pred_classes, classes = imbalance_strategy(pred_classes, classes)
            class_loss = class_loss_func(
                    pred_classes,
                    classes,
                    size_average=False) / nb_pos
            is_train = ((batch_index % 10) > 0 or len(dataloader) < 10)
            detector.zero_grad()
            loss = loc_loss + class_loss
            loss.backward()
            # eval mini-batch
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
            if nb_iter % 10 == 0 or True:
                idx = np.random.randint(0, len(images))
                orig_img = load_image(filenames[idx])
                orig_img = resize(orig_img, (768, 1024), preserve_range=True)
                orig_h, orig_w = orig_img.shape[:2]
                true_classes = classes_orig[idx].long().cpu().numpy()
                true_boxes = encoded_boxes_orig[idx].view(-1, 4)
                true_boxes = true_boxes.cpu().numpy()
                true_boxes = decode_bounding_boxes(
                    anchors, true_boxes
                )
                true_boxes[:, XMIN] *= orig_w
                true_boxes[:, YMIN] *= orig_h
                true_boxes[:, WIDTH] *= orig_w
                true_boxes[:, HEIGHT] *= orig_h

                pred_classes = nn.Softmax(dim=2)(pred_classes_orig)[idx]
                pred_classes = pred_classes.view(-1, detector.nb_classes)
                pred_classes = pred_classes.detach().cpu().numpy()

                pred_boxes = pred_boxes_orig[idx].view(-1, 4)
                pred_boxes = pred_boxes.detach().cpu().numpy()
                pred_boxes = decode_bounding_boxes(
                    anchors, pred_boxes)
                pred_boxes[:, XMIN] *= orig_w
                pred_boxes[:, YMIN] *= orig_h
                pred_boxes[:, WIDTH] *= orig_w
                pred_boxes[:, HEIGHT] *= orig_h

                pred_scores = pred_classes.max(axis=1)
                pred_classes = pred_classes.argmax(axis=1)
                keep = (pred_classes > 0)
                pred_classes = pred_classes[keep]
                pred_scores = pred_scores[keep]
                pred_boxes = pred_boxes[keep]

                keep = true_classes > 0
                true_classes = true_classes[keep]
                true_boxes = true_boxes[keep]

                true_classes = [dataset.decode_class[class_id]
                                for class_id in true_classes]
                pred_classes = [dataset.decode_class[class_id]
                                for class_id in pred_classes]
                orig_img = draw_bounding_boxes(
                    orig_img,
                    true_boxes,
                    true_classes,
                    color=(255, 0, 0),
                    text_color=(255, 0, 0),
                )
                orig_img = draw_bounding_boxes(
                    orig_img,
                    pred_boxes,
                    pred_classes,
                    scores=pred_scores,
                    color=(0, 255, 0),
                    text_color=(0, 255, 0),
                )
                orig_img = orig_img.astype('uint8')
                orig_img = orig_img.transpose((2, 0, 1))
                writer.add_image('data/image', orig_img, nb_iter)
            if nb_iter % 100 == 0:
                # Saving
                model_dict['nb_iter'] = nb_iter
                print('Saving...')
                torch.save(model_dict, 'model.th')
            nb_iter += 1


if __name__ == '__main__':
    run(main)
