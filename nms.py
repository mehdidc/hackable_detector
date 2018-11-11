import numpy as np
from match import iou

def non_maximal_suppression(boxes, confidences, thres=0.3):
    # TODO replace with cv2.dnn.NMSBoxes ?
    boxes = np.array(boxes)
    confidences = np.array(confidences)
    indices = np.argsort(confidences)[::-1]
    boxes = boxes[indices]
    confidences = confidences[indices]
    boxes = np.array(boxes)
    confidences = np.array(confidences)
    indices = np.array(indices)

    result_indices = []
    while len(boxes) > 0:
        cur_box = boxes[0]
        cur_confidence = confidences[0]
        cur_index = indices[0]
        result_indices.append(cur_index)

        next_boxes = boxes[1:]
        next_confidences = confidences[1:]
        next_indices = indices[1:]

        cur_box = cur_box.reshape((1, 4))

        ious = iou(cur_box, next_boxes)
        boxes = next_boxes[ious < thres]
        confidences = next_confidences[ious < thres]
        indices = next_indices[ious < thres]
    return result_indices
