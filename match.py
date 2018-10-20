import numpy as np
from bounding_boxes import get_boxes_coords


def match(anchor_boxes, boxes, iou_threshold=0.5):
    ious = iou_all_pairs(anchor_boxes, boxes)
    matching = np.zeros_like(ious).astype('bool')
    matching[np.argmax(ious, axis=0), np.arange(len(boxes))] = 1
    matching[np.arange(len(anchor_boxes)), np.argmax(ious, axis=1)] = np.max(ious, axis=1) > iou_threshold
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
