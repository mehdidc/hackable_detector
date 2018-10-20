import numpy as np
import torch.nn as nn
import torch
from bounding_boxes import get_boxes_coords
from bounding_boxes import XMIN, YMIN, WIDTH, HEIGHT

class Anchors:

    def __init__(self, scales, aspect_ratios_for_scale, feature_map_dim_for_scale, offset=0.5):
        self.scales = scales
        self.aspect_ratios_for_scale = aspect_ratios_for_scale
        self.feature_map_dim_for_scale = feature_map_dim_for_scale
        self.offset = offset
    
    def bounding_boxes(self):
        boxes = self.bounding_boxes_on_grid()
        boxes = [boxes_for_scale.reshape((-1, 4)) for boxes_for_scale in boxes]
        return np.concatenate(boxes, axis=0)

    def bounding_boxes_on_grid(self):
        boxes = []
        for scale, aspect_ratios, feature_map_dim in zip(self.scales, self.aspect_ratios_for_scale, self.feature_map_dim_for_scale):
            boxes_for_scale = np.empty((feature_map_dim, feature_map_dim, len(aspect_ratios), 4))
            for i, aspect_ratio in enumerate(aspect_ratios):
                y, x = np.indices((feature_map_dim, feature_map_dim))
                boxes_for_scale[:, :, i, XMIN] = (x + self.offset) / feature_map_dim
                boxes_for_scale[:, :, i, YMIN] = (y + self.offset) / feature_map_dim
                boxes_for_scale[:, :, i, WIDTH] = scale * np.sqrt(aspect_ratio)
                boxes_for_scale[:, :, i, HEIGHT] = scale / np.sqrt(aspect_ratio)
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
            scale = detector.scales[i]
            feature_map_dim_for_scale.append(feature_map_dim)
        # create Anchors instance
        return Anchors(
            detector.scales, 
            detector.aspect_ratios_for_scale, 
            feature_map_dim_for_scale, 
            offset=offset
        )
