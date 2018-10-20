import numpy as np
import torch
import torch.nn as nn
from anchors import Anchors
from match import match
from bounding_boxes import encode_bounding_boxes


if __name__ == '__main__':
    from torchvision.models import vgg16
    from detector import DetectorBuilder
    builder = DetectorBuilder()
    builder.add_layers_from(vgg16().features)
    builder.add_layers([
        nn.Conv2d(512, 128, kernel_size=3, padding=1)
    ])
    builder.add_prediction_branch(scale=0.1, aspect_ratios=[1, 1/2])
    builder.add_layers([
        nn.Conv2d(128, 128, kernel_size=3, padding=1)
    ])
    detector = builder.build(nb_classes=10)
    anchors = Anchors.from_detector(
        detector, input_shape=(3, 224, 224))


    nb_examples = 10
    encoded_boxes_for_example = []
    classes_for_example = []
    for i in range(nb_examples):
        anchor_boxes = anchors.bounding_boxes()
        true_boxes = np.random.uniform(size=(5, 4))
        true_classes = np.random.randint(1, 10, size=(5,))

        matching = match(anchor_boxes, true_boxes)
        rows, cols = np.where(matching)
        
        encoded_boxes = np.zeros_like(anchor_boxes)
        encoded_boxes[rows] = encode_bounding_boxes(
            anchor_boxes[rows], 
            true_boxes[cols]
        )
        encoded_boxes = torch.from_numpy(encoded_boxes)
        encoded_boxes = encoded_boxes.float()
        encoded_boxes_for_example.append(encoded_boxes)
        
        classes = np.zeros((len(anchor_boxes),))
        classes[rows] = true_classes[cols]
        classes = torch.from_numpy(classes)
        classes = classes.long()
        classes_for_example.append(classes)

    true_encoded_boxes = torch.stack(encoded_boxes_for_example, dim=0)
    true_encoded_boxes = true_encoded_boxes.view(-1, 4)

    true_classes = torch.stack(classes_for_example, dim=0)
    true_classes = true_classes.view(-1)

    x = torch.zeros(10, 3, 224, 224)
    pred_encoded_boxes, pred_classes = detector.predict(x)
    print(pred_encoded_boxes.size(), true_encoded_boxes.size())
    print(pred_classes.size(), true_classes.size())
