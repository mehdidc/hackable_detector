import numpy as np
import torch
from bounding_boxes import get_boxes_coords
from bounding_boxes import XMIN, YMIN, WIDTH, HEIGHT
from bounding_boxes import encode_bounding_boxes
from match import match_ssd_method


def build_anchors_matrix(
    scales, per_scale_aspect_ratios, per_scale_feature_maps, offset=0.5
):
    boxes = build_anchors_grid_per_scale(
        scales, per_scale_aspect_ratios, per_scale_feature_maps, offset=offset
    )
    boxes = [boxes_for_scale.reshape((-1, 4)) for boxes_for_scale in boxes]
    return np.concatenate(boxes, axis=0)


def build_anchors_grid_per_scale(
    scales, per_scale_aspect_ratios, per_scale_feature_maps, offset=0.5
):
    boxes = []
    for scale, aspect_ratios, feature_map_dim in zip(
        scales, per_scale_aspect_ratios, per_scale_feature_maps
    ):
        boxes_for_scale = np.empty(
            (feature_map_dim, feature_map_dim, len(aspect_ratios), 4)
        )
        for i, aspect_ratio in enumerate(aspect_ratios):
            y, x = np.indices((feature_map_dim, feature_map_dim))
            boxes_for_scale[:, :, i, XMIN] = (x + offset) / feature_map_dim
            boxes_for_scale[:, :, i, YMIN] = (y + offset) / feature_map_dim
            boxes_for_scale[:, :, i, WIDTH] = scale * np.sqrt(aspect_ratio)
            boxes_for_scale[:, :, i, HEIGHT] = scale / np.sqrt(aspect_ratio)
        boxes.append(boxes_for_scale)
    return boxes


def build_anchors_matrix_from_detector(detector, input_shape, offset=0.5):
    # extract feature map dim per scale from the `detector`
    shape = (1,) + input_shape
    x = torch.zeros(*shape)
    bbox_outputs, _ = detector.predict_raw(x)
    feature_map_dim_for_scale = []
    for i, bbox_output in enumerate(bbox_outputs):
        _, _, feature_map_dim, _ = bbox_output.size()
        feature_map_dim_for_scale.append(feature_map_dim)
    return build_anchors_matrix(
        detector.scales,
        detector.aspect_ratios_for_scale,
        feature_map_dim_for_scale,
        offset=offset,
    )


def match_and_encode(anchor_boxes, boxes, classes, match_method=match_ssd_method):
    if len(boxes) == 0:
        return np.zeros_like(anchor_boxes), np.zeros((anchor_boxes.shape[0],))
    boxes = np.array(boxes)
    classes = np.array(classes)

    matching = match_method(anchor_boxes, boxes)
    rows, cols = np.where(matching)
    encoded_boxes = np.zeros_like(anchor_boxes)
    encoded_boxes[rows] = encode_bounding_boxes(anchor_boxes[rows], boxes[cols])
    encoded_boxes = torch.from_numpy(encoded_boxes)
    encoded_boxes = encoded_boxes.float()
    encoded_classes = np.zeros((len(anchor_boxes),))
    encoded_classes[rows] = classes[cols]
    if (encoded_classes > 0).sum() < len(classes):
        import pdb

        pdb.set_trace()
    encoded_classes = torch.from_numpy(encoded_classes)
    encoded_classes = encoded_classes.long()
    return encoded_boxes, encoded_classes
