import os
import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
from data.base import DetectionDataset
from data.coco import get_annotations
from visualization import draw_bounding_boxes
from bounding_boxes import decode_bounding_boxes
from bounding_boxes import encode_bounding_boxes
from bounding_boxes import XMIN, YMIN, WIDTH, HEIGHT
from nms import non_maximal_suppression
from match import match_ordered_boxes, match
from albumentations import HorizontalFlip

from skimage.io import imsave, imread
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from albumentations import Compose
from albumentations import Normalize
from albumentations import Resize

from train import prediction_layer

data = torch.load('model.th')
model = data['model']
model.cuda()
model.eval()

anchors = data['anchors']
transform = data['transform']

filenames, annotations = get_annotations(
    images_folder='small',
    annotations_file='small/annotations.json'
)
image_size = 224 
transform = Compose([
    HorizontalFlip(p=0.0),
    Resize(height=image_size, width=image_size, p=1.0),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.),
])
dataset = DetectionDataset(
    filenames, 
    annotations,
    transform=transform,
)
for idx in range(100):
    image, true_boxes, true_classes = dataset[idx]
    print(filenames[idx])
    orig_img = imread(filenames[idx])
    if len(orig_img.shape) == 2:
        orig_img = orig_img[:, :, np.newaxis]
        orig_img = orig_img * np.ones((1, 1, 3))
    elif len(orig_img.shape) == 1:
        orig_img = orig_img[:, np.newaxis, np.newaxis]
        orig_img = orig_img * np.ones((1, orig_img.shape[0], 3))
    orig_h, orig_w, _ = orig_img.shape

    true_classes = true_classes.long().numpy()
    true_boxes = true_boxes.numpy()
    true_boxes[:, XMIN] *=  orig_w
    true_boxes[:, YMIN] *=  orig_h
    true_boxes[:, WIDTH] *=  orig_w
    true_boxes[:, HEIGHT] *=  orig_h

    image = image.view((1,) + image.size())
    image = image.float().cuda()
    with torch.no_grad():
        pred_boxes, pred_classes = model.predict(image)
    pred_classes = nn.Softmax(dim=2)(pred_classes)
    pred_boxes = pred_boxes.view(-1, 4)
    pred_classes = pred_classes.view(-1, model.nb_classes)
    pred_boxes = pred_boxes.cpu().numpy()
    pred_classes = pred_classes.cpu().numpy()

    pred_boxes = decode_bounding_boxes(anchors.bounding_boxes, pred_boxes)
    #pred_boxes = anchors.bounding_boxes.copy()
    pred_boxes[:, XMIN] *= orig_w
    pred_boxes[:, YMIN] *=  orig_h
    pred_boxes[:, WIDTH] *=  orig_w
    pred_boxes[:, HEIGHT] *=  orig_h

    # eval
    evaluate = False
    if evaluate:
        for class_id in np.unique(true_classes):
            pred_scores_class = pred_classes[:, class_id]
            #indices = non_maximal_suppression(pred_boxes, pred_scores_class)
            #pred_boxes_class = pred_boxes[indices]
            #pred_scores_class = pred_scores_class[indices]

            indices = np.argsort(-pred_scores_class)
            print(pred_boxes.shape, pred_scores_class.shape)
            pred_boxes_class = pred_boxes[indices]
            pred_scores_class = pred_scores_class[indices]
            
            true_boxes_class = true_boxes[true_classes==class_id]

            matching = match_ordered_boxes(pred_boxes_class, true_boxes_class, iou_threshold=0.5)
            true = np.any(matching, axis=1).astype('int32')
            title = 'Class {}'.format(dataset.decode_class[class_id])
            print(title)
            print('#' * len(title))
            for thresh in np.linspace(0.01, 0.99, 10):
                print('Threshold : {:.3f}'.format(thresh))
                pred = (pred_scores_class >= thresh).astype('int32')
                nb_true_positives = ((pred==1) & (true==1)).sum()
                if pred.sum():
                    precision = (nb_true_positives / pred.sum())
                else:
                    precision = 0
                recall = nb_true_positives / len(true_boxes_class)
                print('Detected : {}/{}'.format(nb_true_positives, len(true_boxes_class)))
                print('Precision : {:.5f}'.format(precision))
                print('')
    # draw anchors
    draw_anchors = True 
    aboxes = anchors.bounding_boxes.copy()
    aboxes[:, XMIN] *= orig_w
    aboxes[:, YMIN] *= orig_h
    aboxes[:, WIDTH] *= orig_w
    aboxes[:, HEIGHT] *= orig_h
    if draw_anchors:
        matching = match(aboxes, true_boxes)
        rows, cols = np.where(matching)
        boxes = aboxes[rows].copy()
        orig_img = draw_bounding_boxes(
            orig_img,
            boxes,
            [''] * len(boxes),
            color=(255, 0, 255), 
            text_color=(255, 0, 255),
        )
    # viz
    pred_scores = pred_classes.max(axis=1)
    pred_classes = pred_classes.argmax(axis=1)
    keep = (pred_classes > 0) #& (pred_scores > 0.99)
    pred_classes = pred_classes[keep]
    pred_scores = pred_scores[keep]
    pred_boxes = pred_boxes[keep]
    """
    mask = match(aboxes, true_boxes)
    rows, cols = np.where(mask)
    pred_boxes = pred_boxes[rows]
    pred_scores = pred_classes[rows, 1:].max(axis=1)
    pred_classes = pred_classes[rows, 1:].argmax(axis=1) + 1
    """

    """
    indices = non_maximal_suppression(pred_boxes, pred_scores)
    pred_boxes = pred_boxes[indices]
    pred_scores = pred_scores[indices]
    pred_classes = pred_classes[indices]
    """
    true_classes = [dataset.decode_class[class_id] for class_id in true_classes]
    pred_classes = [dataset.decode_class[class_id] for class_id in pred_classes]
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
    imsave('test/' + os.path.basename(filenames[idx]), orig_img)
