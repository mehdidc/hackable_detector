import numpy as np
import torch
from bounding_boxes import get_boxes_coords
from bounding_boxes import XMIN, YMIN, WIDTH, HEIGHT
from bounding_boxes import encode_bounding_boxes
from match import match_ssd_method


class Anchors:

    def __init__(self, scales, aspect_ratios_for_scale, feature_map_dim_for_scale, offset=0.5):
        self.scales = scales
        self.aspect_ratios_for_scale = aspect_ratios_for_scale
        self.feature_map_dim_for_scale = feature_map_dim_for_scale
        self.offset = offset
        self._prepare()

    def _prepare(self):
        self.bounding_boxes = self._build_bounding_boxes()

    def _build_bounding_boxes(self):
        boxes = self.bounding_boxes_on_grid()
        boxes = [boxes_for_scale.reshape((-1, 4)) for boxes_for_scale in boxes]
        return np.concatenate(boxes, axis=0)

    def bounding_boxes_on_grid(self):
        boxes = []
        for scale, aspect_ratios, feature_map_dim in zip(self.scales, self.aspect_ratios_for_scale, self.feature_map_dim_for_scale):
            boxes_for_scale = np.empty(
                (feature_map_dim, feature_map_dim, len(aspect_ratios), 4))
            for i, aspect_ratio in enumerate(aspect_ratios):
                y, x = np.indices((feature_map_dim, feature_map_dim))
                boxes_for_scale[:, :, i, XMIN] = (
                    x + self.offset) / feature_map_dim
                boxes_for_scale[:, :, i, YMIN] = (
                    y + self.offset) / feature_map_dim
                boxes_for_scale[:, :, i, WIDTH] = scale * np.sqrt(aspect_ratio)
                boxes_for_scale[:, :, i, HEIGHT] = scale / \
                    np.sqrt(aspect_ratio)
            boxes.append(boxes_for_scale)
        return boxes

    @staticmethod
    def from_detector(detector, input_shape=(3, 224, 224), offset=0.5):
        # extract feature map dim per scale from the `detector`
        shape = (1,) + input_shape
        x = torch.zeros(*shape)
        bbox_outputs, _ = detector.predict_raw(x)
        feature_map_dim_for_scale = []
        for i, bbox_output in enumerate(bbox_outputs):
            _, _, feature_map_dim, _ = bbox_output.size()
            feature_map_dim_for_scale.append(feature_map_dim)
        # create Anchors instance
        return Anchors(
            detector.scales,
            detector.aspect_ratios_for_scale,
            feature_map_dim_for_scale,
            offset=offset
        )

    def match_and_encode(self, image_boxes, image_classes, match_method=match_ssd_method):
        anchor_boxes = self.bounding_boxes
        if len(image_boxes) == 0:
            return np.zeros_like(anchor_boxes), np.zeros((anchor_boxes.shape[0],))
        image_boxes = np.array(image_boxes)
        image_classes = np.array(image_classes)

        matching = match_method(anchor_boxes, image_boxes)
        rows, cols = np.where(matching)
        encoded_boxes = np.zeros_like(anchor_boxes)
        encoded_boxes[rows] = encode_bounding_boxes(
            anchor_boxes[rows],
            image_boxes[cols]
        )
        encoded_boxes = torch.from_numpy(encoded_boxes)
        encoded_boxes = encoded_boxes.float()
        classes = np.zeros((len(anchor_boxes),))
        classes[rows] = image_classes[cols]
        if (classes > 0).sum() < len(image_classes):
            import pdb
            pdb.set_trace()
        classes = torch.from_numpy(classes)
        classes = classes.long()
        return encoded_boxes, classes
