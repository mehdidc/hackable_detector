import torch
import numpy as np
from bounding_boxes import get_boxes_coords
import numba


def match_ssd_method(anchor_boxes, boxes, iou_threshold=0.5):
    ious = iou_all_pairs(anchor_boxes, boxes)
    matching = np.zeros_like(ious).astype('bool')
    matching[np.argmax(ious, axis=0), np.arange(len(boxes))] = True
    matching[np.arange(len(anchor_boxes)), np.argmax(ious, axis=1)] |= (np.max(ious, axis=1) > iou_threshold)
    return matching

def match_bijective_method(anchor_boxes, boxes, iou_threshold=0.5):
    ious = iou_all_pairs(anchor_boxes, boxes)
    matching = np.zeros_like(ious).astype('bool')
    area = boxes[:, 2] * boxes[:, 3]
    for b in np.argsort(area)[::-1]:
        alist = np.argsort(ious[:, b])[::-1]
        for a in alist:
            if np.all(~matching[a]):
                matching[a, b] = True
                break
    return matching


def iou_all_pairs(boxes, other_boxes):
    return iou(
        boxes.reshape((len(boxes), 1, 4)),
        other_boxes.reshape((1, len(other_boxes), 4)),
    )


def iou(boxes, other_boxes, eps=1e-10):
    ax, ay, aw, ah = get_boxes_coords(boxes)
    bx, by, bw, bh = get_boxes_coords(other_boxes)

    xmin = np.maximum(ax, bx)
    ymin = np.maximum(ay, by)
    xmax = np.minimum(ax + aw, bx + bw)
    ymax = np.minimum(ay + ah, by + bh)

    w_intersection = np.clip(xmax - xmin, a_min=0, a_max=None)
    h_intersection = np.clip(ymax - ymin, a_min=0, a_max=None)
    intersection = w_intersection * h_intersection
    union = aw * ah + bw * bh - intersection
    return intersection / (union + eps)


def match_ordered_boxes(pred_boxes, true_boxes, iou_threshold=0.5):
    ious = iou_all_pairs(pred_boxes, true_boxes)
    matching = np.zeros_like(ious).astype('bool')
    true_already_matched = np.zeros(len(true_boxes)).astype('bool')
    for pred_ind in range(len(pred_boxes)):
        true_match_ind = -1
        true_best_iou = 0 
        for true_ind in range(len(true_boxes)):
            iou = ious[pred_ind, true_ind]
            if iou <= iou_threshold:
                continue
            if true_already_matched[true_ind]:
                continue
            if iou > true_best_iou:
                true_match_ind = true_ind
                true_best_iou = iou
        if true_match_ind < 0:
            continue
        true_already_matched[true_match_ind] = True
        matching[pred_ind, true_match_ind] = True
    return matching
