import numpy as np

XMIN, YMIN, WIDTH, HEIGHT = 0, 1, 2, 3

def encode_bounding_boxes(anchors_boxes, image_boxes, eps=1e-10):
    ax, ay, aw, ah = get_boxes_coords(anchors_boxes)
    bx, by, bw, bh = get_boxes_coords(image_boxes)
    boxes = np.zeros_like(anchors_boxes)
    boxes[:, XMIN] = (bx - ax) / aw
    boxes[:, YMIN] = (by - by) / ah
    boxes[:, WIDTH] = np.log(eps + bw / aw)
    boxes[:, HEIGHT] = np.log(eps + bh / ah)
    return boxes

def get_boxes_coords(boxes):
    x, y, w, h = (
        boxes[..., XMIN], 
        boxes[..., YMIN], 
        boxes[..., WIDTH], 
        boxes[..., HEIGHT]
    )
    return x, y, w, h
