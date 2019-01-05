import numpy as np

XMIN, YMIN, WIDTH, HEIGHT = 0, 1, 2, 3


def encode_bounding_boxes(anchors_boxes, image_boxes, eps=1e-10):
    ax, ay, aw, ah = get_boxes_coords(anchors_boxes)
    bx, by, bw, bh = get_boxes_coords(image_boxes)
    boxes = np.zeros_like(anchors_boxes)
    assert np.all(bw > 0)
    assert np.all(aw > 0)
    boxes[:, XMIN] = (bx - ax) / aw
    boxes[:, YMIN] = (by - ay) / ah
    boxes[:, WIDTH] = np.log(eps + bw / aw)
    boxes[:, HEIGHT] = np.log(eps + bh / ah)
    return boxes


def decode_bounding_boxes(anchors_boxes, pred_boxes, eps=1e-10):
    ax, ay, aw, ah = get_boxes_coords(anchors_boxes)
    bx, by, bw, bh = get_boxes_coords(pred_boxes)
    boxes = np.zeros_like(anchors_boxes)
    boxes[:, XMIN] = (bx * aw) + ax
    boxes[:, YMIN] = (by * ah) + ay
    boxes[:, WIDTH] = np.exp(bw) * aw
    boxes[:, HEIGHT] = np.exp(bh) * ah
    return boxes


def get_boxes_coords(boxes):
    x, y, w, h = (
        boxes[..., XMIN],
        boxes[..., YMIN],
        boxes[..., WIDTH],
        boxes[..., HEIGHT],
    )
    return x, y, w, h


def scale_boxes(boxes, scale_w, scale_h):
    return [
        (x * scale_w, y * scale_h, w * scale_w, h * scale_h) for x, y, w, h in boxes
    ]


def center_boxes(boxes):
    return [(x + w / 2, y + h / 2, w, h) for x, y, w, h in boxes]


def uncenter_boxes(boxes):
    return [(x - w / 2, y - h / 2, w, h) for x, y, w, h in boxes]


def boxes_width_height_to_min_max_format(boxes):
    return [(x, y, x + w, y + h) for x, y, w, h in boxes]


def boxes_min_max_to_width_height_format(boxes):
    return [(xmin, ymin, xmax - xmin, ymax - ymin) for xmin, ymin, xmax, ymax in boxes]
