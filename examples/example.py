import cv2
import sys
sys.path.append('..')
from data.base import DetectionDataset
from data.coco import get_annotations
from visualization import draw_bounding_boxes
from torchvision.transforms import Compose

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose
)

transform = Compose([])
filenames, annotations = get_annotations(
    images_folder='.',
    annotations_file='annotations/instances_val2014.json'
)
dataset = DetectionDataset(
    filenames, 
    annotations, 
    transform=transform,
)
image, boxes, classes = dataset[0]
classes = [str(class_name) for class_name in classes]
image = image.numpy()
image = draw_bounding_boxes(image, boxes, classes)
cv2.imwrite('out.png', image)
