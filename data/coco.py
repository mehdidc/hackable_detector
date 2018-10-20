import os
from pycocotools.coco import COCO as CocoAPI
from collections import defaultdict


def get_annotations(images_folder, annotations_file):
    api = CocoAPI(annotations_file)
    image_annotations = defaultdict(list)
    category_name = {}
    for category_id, cat in api.cats.items():
        category_name[category_id] = cat['name']
    image_filename = {}
    for ann in api.loadAnns(api.getAnnIds()):
        image_id = ann['image_id']
        box = ann['bbox']
        category = category_name[ann['category_id']]
        image_annotations[image_id].append((box, category))
    for im in api.loadImgs(api.getImgIds()):
        image_id = im['id']
        filename = os.path.join(images_folder, im['file_name'])
        image_filename[image_id] = filename

    image_ids = list(image_filename.keys())
    filenames = []
    annotations = []
    for image_id in image_ids:
        filenames.append(image_filename[image_id])
        annotations.append(image_annotations[image_id])
    return filenames, annotations

if __name__ == '__main__':
    filenames, annotations = get_annotations('ddwd', 'annotations/instances_val2014.json')
    print(annotations[0:10])
