#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import sys
import uuid

import numpy as np
import PIL.Image

import labelme
import pycocotools.mask


OUT_DIR = "./coco_labelme"
ANNOT_DIR = os.path.join(OUT_DIR, 'annotations')

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('raw_labelme_dir', help='Path to input `raw_labelme` directory')
    args = parser.parse_args()

    if os.path.exists(OUT_DIR):
        print('Output directory already exists:' + OUT_DIR)
        sys.exit(1)

    labels_filepath = os.path.join(args.raw_labelme_dir, 'labels.txt')
    if not os.path.exists(labels_filepath):
        print('Labels file does not exist')
        sys.exit(1)

    print('Creating dataset:', OUT_DIR)
    os.makedirs(OUT_DIR)
    os.makedirs(ANNOT_DIR)

    partitions = ["train", "val"]
    now = datetime.datetime.now()

    # Run for "train" and "val" partitions.
    for partition in partitions:
        IMG_DIR_IN = os.path.join(args.raw_labelme_dir, f"{partition}_images")
        IMG_DIR_OUT = os.path.join(OUT_DIR, f"{partition}2014")

        ANN_DIR_IN = os.path.join(args.raw_labelme_dir, f"{partition}_annotations")
        ANN_FILEPATH = os.path.join(ANNOT_DIR, f"instances_{partition}2014.json")

        os.makedirs(IMG_DIR_OUT)

        class_name_to_id = {}

        data = dict(
            info=dict(
                description=None,
                url=None,
                version=None,
                year=now.year,
                contributor=None,
                date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
            ),
            licenses=[dict(
                url=None,
                id=0,
                name=None,
            )],
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            type='instances',
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=[
                # supercategory, id, name
            ],
        )

        for class_id, line in enumerate(open(labels_filepath).readlines()):
            class_name = line.strip()
            if class_id == 0:
                assert class_name == '__ignore__'
                continue
            class_name_to_id[class_name] = class_id
            data['categories'].append(dict(
                supercategory=None,
                id=class_id,
                name=class_name,
                ))



        label_files = glob.glob(os.path.join(ANN_DIR_IN, '*.json'))
        for image_id, filename in enumerate(label_files):
            print('Generating dataset from:', filename)

            label_file = labelme.LabelFile(filename=filename)
            base = os.path.splitext(os.path.basename(filename))[0]
            out_img_file = os.path.join(IMG_DIR_OUT, base + ".jpg")

            img = labelme.utils.img_data_to_arr(label_file.imageData)
            PIL.Image.fromarray(img).save(out_img_file)
            data['images'].append(dict(
                license=0,
                url=None,
                file_name=os.path.basename(out_img_file),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            ))

            masks = {}                                     # for area
            segmentations = collections.defaultdict(list)  # for segmentation
            for shape in label_file.shapes:
                points = shape['points']
                label = shape['label']
                group_id = shape.get('group_id')
                shape_type = shape.get('shape_type')
                mask = labelme.utils.shape_to_mask(
                    img.shape[:2], points, shape_type
                )

                if group_id is None:
                    group_id = uuid.uuid1()

                instance = (label, group_id)

                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask

                points = np.asarray(points).flatten().tolist()
                segmentations[instance].append(points)
            segmentations = dict(segmentations)

            for instance, mask in masks.items():
                cls_name, group_id = instance
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]

                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                data['annotations'].append(dict(
                    id=len(data['annotations']),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                ))

        with open(ANN_FILEPATH, 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    main()
