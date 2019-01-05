import cv2
import numpy as np


def draw_bounding_boxes(
    image,
    bounding_boxes,
    classes,
    scores=None,
    color=(0, 255, 0),
    text_color=(0, 255, 0),
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale=1.0,
    pad=0,
):

    for i, (bounding_box, class_name) in enumerate(zip(bounding_boxes, classes)):
        x, y, w, h = bounding_box
        xmin = (x) + pad
        ymin = (y) + pad
        xmax = (x + w) + pad
        ymax = (y + h) + pad
        xmin = np.clip(xmin, 0, image.shape[1])
        xmax = np.clip(xmax, 0, image.shape[1])
        ymin = np.clip(ymin, 0, image.shape[0])
        ymax = np.clip(ymax, 0, image.shape[0])
        xmin = int(round(xmin))
        ymin = int(round(ymin))
        xmax = int(round(xmax))
        ymax = int(round(ymax))
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color)
        if scores is not None:
            text = "{}({:.2f})".format(class_name, scores[i])
        else:
            text = class_name
        image = cv2.putText(image, text, (xmin, ymin), font, font_scale, text_color, 2)
    return image
