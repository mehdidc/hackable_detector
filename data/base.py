import numpy as np
import torch
from torchvision.datasets.folder import default_loader
from skimage.io import imread

class DetectionDataset:

    def __init__(self, filenames, annotations, transform=None):
        self.filenames = filenames
        self.annotations = annotations
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
        classes = [self.encode_class[class_id] for box, class_id in annotations]
        filename = 'dog.png'
        im = imread(filename)
        if self.transform:
            annotations = {
                'image': im, 
                'bboxes': boxes, 
                'category_id': classes,
            }
            annotations = self.transform(**annotations)
            im = annotations['image']
            boxes = annotations['bboxes']
            classes = annotations['category_id']
        im = torch.from_numpy(im)
        return im, boxes, classes
