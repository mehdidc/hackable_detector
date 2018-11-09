import numpy as np
import torch
from torchvision.datasets.folder import default_loader
from imageio import imread

from bounding_boxes import boxes_min_max_to_width_height_format
from bounding_boxes import boxes_width_height_to_min_max_format
from bounding_boxes import scale_boxes


class DetectionDataset:

    def __init__(self, filenames, annotations, encode_boxes=False, anchors=None, match_method=None, transform=None):
        if encode_boxes:
            assert anchors and match_method, 'If encode_boxes is True, you need to provide `anchors` and `match_methdod`'
        self.filenames = filenames
        self.annotations = annotations
        self.encode_boxes = encode_boxes
        self.anchors = anchors
        self.match_method = match_method
        self.transform = transform
        self._prepare()

    def _prepare(self):
        classes = [
            class_id for anns in self.annotations for box, class_id in anns]
        classes = np.unique(classes)
        classes = sorted(classes)
        self.encode_class = {
            class_id: (i + 1)
            for i, class_id in enumerate(classes)
        }
        self.decode_class = {
            (i + 1): class_id
            for i, class_id in enumerate(classes)
        }

    def __getitem__(self, index):
        filename = self.filenames[index]
        annotations = self.annotations[index]
        boxes = [box for box, class_id in annotations]
        classes = [self.encode_class[class_id]
                   for box, class_id in annotations]
        im = load_image(filename)
        image_height, image_width, _ = im.shape
        boxes = scale_boxes(boxes, 1 / image_width, 1 / image_height)
        if self.transform:
            boxes = boxes_width_height_to_min_max_format(boxes)
            annotations = {
                'image': im,
                'bboxes': boxes,
                'category_id': classes,
            }
            annotations = self.transform(**annotations)
            im = annotations['image']
            boxes = annotations['bboxes']
            classes = annotations['category_id']
            boxes = boxes_min_max_to_width_height_format(boxes)
        im = im.transpose((2, 0, 1))
        if self.encode_boxes:
            boxes, classes = self.anchors.match_and_encode(
                boxes, classes,
                match_method=self.match_method,
            )
        im = torch.from_numpy(im).float()
        boxes = torch.from_numpy(np.array(boxes)).float()
        classes = torch.from_numpy(np.array(classes)).float()
        return im, boxes, classes, filename

    def __len__(self):
        return len(self.filenames)


def load_image(filename):
    im = imread(filename)
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = im * np.ones((1, 1, 3))
    elif len(im.shape) == 1:
        im = im[:, np.newaxis, np.newaxis]
        im = im * np.ones((1, im.shape[0], 3))
    return im
