"""
Helper functions and utilities to preprocess the ADE20K dataset.

@author Osama Sakhi
"""

from collections import defaultdict
import shutil
import json
import os
import re
from datetime import datetime

from pathlib import Path
from PIL import Image

import numpy as np
import scipy.io
from imageio import imread, imsave

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cv2

from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi


def filter_ade20k_dataset(src_dir, required_keywords=[], reject_keywords=[]):
    """
    This function filters out unwanted scenes from the ADE20k dataset. A
    dictionary is returned containing the samples from the training and
    validation sets that contains one of the scenes from `required_keywords`
    and has none of the scenes from `reject_keywords`.


    Args:
        :src_dir: - The path to the unzipped ADE20K dataset
        :required_keywords: - A list of scenes/keywords, the samples kept from
            the ADE20K dataset must belong to one of these scenes.
        :reject_keywords: - A list of scenes/scene-subclasses, the samples kept
            from the ADE20k dataset must not belong to any of these. This is
            helpful for certain scenes like "garage" which may have the subsets
            "garage/indoor" and "garage/outdoor" and you want to filter out
            "outdoor".

    Returns:
        :metadata: - A dictionary with keys "training" and "validation", and
            corresponding values are a list of that partition's kept samples.
            The samples will be in the form (img_filepath, seg_filepath).
    """
    metadata = defaultdict(list)
    partitions = ["training", "validation"]

    for partition in partitions:
        partition_root = os.path.join(src_dir, "images", partition)
        root = Path(partition_root)

        for idx, filepath in enumerate(root.glob("**/*.jpg")):
            if filepath.is_dir():
                continue

            # We want to only scan the filepath past the root directory of
            # the partition.
            searchable_name = str(filepath).replace(str(root), "")

            # If not a single required keyword match comes up for this image,
            # skip it.
            req_matches = [kw in searchable_name for kw in required_keywords]
            if len(required_keywords) and not any(req_matches):
                continue

            # If any reject_keyword comes up for this image, skip it.
            reject_matches = [kw in searchable_name for kw in reject_keywords]
            if len(reject_keywords) and any(reject_matches):
                continue

            # Otherwise, record the image and segmentation filepaths into the
            # dataset metadata.
            img_filepath = str(filepath)
            seg_filepath = img_filepath.replace(".jpg", "_seg.png")

            sample_metadata = (img_filepath, seg_filepath)
            metadata[partition].append(sample_metadata)

    return metadata


def get_all_ade20k_labels(src_dir):
    """
    Given a path to the ADE20K dataset, this function will return
    a list of all of the labels used by ADE20K as a list.
    """
    meta = scipy.io.loadmat(os.path.join(src_dir, 'index_ade20k.mat'), squeeze_me=True)
    # Only the 'index' field stores the data
    meta_index = meta['index']
    # Create a dict of all the metadata fields
    ade_metadata = {name: meta_index[name][()] for name in meta_index.dtype.names}

    # The label names are stored in the 'objectnames' key
    all_labels = ade_metadata['objectnames'].tolist()

    return all_labels


def get_new_label_mappings(src_dir, grouped_labels):
    """
    """
    original_labels = get_all_ade20k_labels(src_dir)

    # Map ALL text labels to integer values
    label_to_old_id = {label: i for (i, label) in enumerate(original_labels)}
    # Create a reverse mapping -- integers to text labels.
    old_id_to_label = {i: label for (label, i) in label_to_old_id.items()}

    # Map individual labels to their new_ids
    label_to_new_id = {}
    # Map new_ids to a string representing the label group.
    new_id_to_label = {}

    # Mape old_ids to new_ids if the original label is kept
    old_id_to_new_id = {}
    # Background label always maps to 0
    old_id_to_new_id[0] = 0

    for new_id, group in enumerate(grouped_labels):
        for label in group:
            label_to_new_id[label] = new_id

            if label in label_to_old_id:
                old_id = label_to_old_id[label]
                old_id_to_new_id[old_id] = new_id

        # The value at new_id will be the group label combined into
        # a single string.
        new_id_to_label[new_id] = ' | '.join(group)


    return (label_to_new_id, new_id_to_label, old_id_to_new_id)

# =============================================================================
#                               VOC Dataset Format
# =============================================================================

def create_new_seg_mask(seg_mask, old_id_to_new_id, new_id_to_label):
    """
    Transform an ADE20K segmentation mask into a VOC-style segmentation
    mask using the mappings old_id_to_new_id and new_id_to_label.
    """

    # Create a new one-channel mask
    h, w, _ = seg_mask.shape
    new_seg_mask = np.zeros((h, w)).astype(np.uint8)

    # The seg_mask's green channel stores the instance id.
    instance_ids = np.unique(seg_mask[:, :, 1])
    for instance_id in instance_ids:
        # Instance id 0 belongs to the background. Since the new_seg_mask is
        # initialized to 0's (background label), we do not have to explicitly
        # handle this case.
        if instance_id == 0:
            continue

        # Get the pixels corresponding to this instance_id.
        pixels = (seg_mask[:, :, 1] == instance_id)

        # The red_channel contains encodes the old_id. We adapt the code from
        # https://groups.csail.mit.edu/vision/datasets/ADE20K/code/loadAde20K.m
        # to extract the old_id (the integer label).
        red_channel_val = seg_mask[pixels][0, 0]
        old_id = red_channel_val // 10 * 256 + instance_id - 1

        # If we're not keeping this category in new dataset, discard it.
        if old_id not in old_id_to_new_id:
            continue

        new_id = old_id_to_new_id[old_id]

        # Color the pixels corresponding to this label with the integer label.
        new_seg_mask[pixels] = new_id

    return new_seg_mask


def is_sparse_mask(new_seg_mask, threshold=.70):
    """
    This informs us if a new_seg_mask has too many "background" pixels.

    If more than the threshold ratio of the image has the integer label
    '0', then we consider the mask to be sparse.

    Args:
        :new_seg_mask: - a 1-channel numpy array
        :threshold: - a value from 0 - 1.0

    Returns:
        :is_sparse: - True if the mask is more than threshold percent
            background pixels, false otherwise
    """
    h, w = new_seg_mask.shape
    n_pixels = h * w
    n_zeros = n_pixels - np.count_nonzero(new_seg_mask)
    ratio = n_zeros / n_pixels
    is_sparse = (ratio >= threshold)

    return is_sparse


def create_voc_directories(dst_dir):
    """Creates the same directory structure as the PASCAL VOC 2012 dataset."""

    VOC_DIR = os.path.join(dst_dir, 'VOCdevkit', 'VOC2012')

    # Create the destination image and segmentation directory if it
    # doesn't exist.
    DST_IMG_DIR = os.path.join(VOC_DIR, 'JPEGImages')
    DST_ANNOT_DIR = os.path.join(VOC_DIR, 'SegmentationClass')
    IMGSET_DIR = os.path.join(VOC_DIR, 'ImageSets')
    PARTITIONS_DIR = os.path.join(IMGSET_DIR, 'Segmentation')

    Path(DST_IMG_DIR).mkdir(parents=True, exist_ok=True)
    Path(DST_ANNOT_DIR).mkdir(parents=True, exist_ok=True)
    Path(IMGSET_DIR).mkdir(parents=True, exist_ok=True)

    # Create the other directories that VOC datasets use, even if we
    # leave them empty.
    Path(os.path.join(VOC_DIR, 'Annotations')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(VOC_DIR, 'SegmentationObject')).mkdir(parents=True, exist_ok=True)

    # These directories are all under the ImageSets Directory
    Path(os.path.join(IMGSET_DIR, 'Action')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(IMGSET_DIR, 'Layout')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(IMGSET_DIR, 'Main')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(IMGSET_DIR, 'Segmentation')).mkdir(parents=True, exist_ok=True)

    return (DST_IMG_DIR, DST_ANNOT_DIR, PARTITIONS_DIR)


def convert_to_voc(src_dir, dst_dir, required_keywords=[],
        reject_keywords=[],
        want_labels=['background',]):

    metadata = filter_ade20k_dataset(src_dir, required_keywords,
            reject_keywords)

    (label_to_new_id, new_id_to_label, \
        old_id_to_new_id) = get_new_label_mappings(src_dir, want_labels)

    (img_dir, annot_dir, partitions_dir) = create_voc_directories(dst_dir)

    for partition, samples in metadata.items():

        # Keep a list of which samples are in the final dataset.
        kept_samples = []

        for (img_fp, seg_fp) in samples:
            sample_name = os.path.basename(img_fp).replace(".jpg", "")
            new_img_fp = os.path.join(img_dir, f"{sample_name}.jpg")
            new_seg_fp = os.path.join(annot_dir, f"{sample_name}.png")

            # Create a new segmentation mask from the original annotation.
            seg_mask = imread(seg_fp)
            new_seg_mask = create_new_seg_mask(seg_mask, old_id_to_new_id, new_id_to_label)

            # We do not want to keep sparse masks in the new dataset.
            if is_sparse_mask(new_seg_mask):
                continue

            # Place the image and the new segmentation mask in the appropriate
            # directories.
            shutil.copy(img_fp, new_img_fp)
            imsave(new_seg_fp, new_seg_mask)

            # Add this sample name to the kept_samples list
            kept_samples.append(sample_name)


        # Now, save the samples to a file.
        if partition == "training":
            partition_fp = os.path.join(partitions_dir, "train.txt")
        else:
            partition_fp = os.path.join(partitions_dir, "val.txt")

        with open(partition_fp, 'w') as wf:
            for sample_name in kept_samples:
                wf.write(sample_name + "\n")

        print(f"Partition {partition} has {len(kept_samples)}.")


    # Store a labelmap mapping labels to new_ids in the destination directory
    # so we can visualize the annotations with corresponding labels later.
    labelmap_fp = os.path.join(dst_dir, 'labelmap.json')
    with open(labelmap_fp, 'w') as wf:
        json.dump(label_to_new_id, wf, indent=4)

    print(f"ADE20K dataset has been converted and stored in {dst_dir}")
    print(f"Label mapping has been stored in {labelmap_fp}")


def get_voc_labelmap(dst_dir):
    """
    Get the label mapping from a VOC dataset. This is the label_to_new_id
    mapping.
    """
    labelmap_fp = os.path.join(dst_dir, 'labelmap.json')
    with open(labelmap_fp, 'r') as rf:
        label_to_new_id = json.load(rf)

    return label_to_new_id

# =============================================================================
#                               COCO Dataset Format
# =============================================================================

def create_coco_directories(dst_dir):
    """Creates the same directory structure as the PASCAL VOC 2012 dataset."""
    trn_dir = os.path.join(dst_dir, "train2014")
    val_dir = os.path.join(dst_dir, "val2014")
    annot_dir = os.path.join(dst_dir, "annotations")

    trn_fpath = os.path.join(annot_dir, "instances_train2014.json")
    val_fpath = os.path.join(annot_dir, "instances_val2014.json")

    # Create the directories
    Path(trn_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)
    Path(annot_dir).mkdir(parents=True, exist_ok=True)

    return (trn_dir, val_dir, trn_fpath, val_fpath)


def create_coco_image_info(img_fp, image_id, width, height):
    """
    Create COCO metadata for each image. Some of the fields specified
    here with constants are required for the format but unused in our
    training and inference pipeline.
    """

    # The sample name is the file's basename without the extension
    basename = os.path.basename(img_fp)
    sample_name, extension = basename.split(".")

    image_info = {
            "id": image_id,
            "file_name": sample_name,
            "width": width,
            "height": height,
            "date_captured": datetime.utcnow().isoformat(' '),
            "license": 1,
            "coco_url": sample_name,
            "flickr_url": ''
    }
    return image_info


def process_coco_sample(img_dir, conversion_metadata, old_id_to_new_id, img_fp, seg_fp):
    local_annotations = []
    next_ann_id = conversion_metadata['next_ann_id']
    next_file_id = conversion_metadata['next_file_id']

    # Read the ADE20k mask.
    seg_mask = imread(seg_fp)

    # The seg_mask's green channel stores the instance id.
    instance_ids = np.unique(seg_mask[:, :, 1])
    for instance_id in instance_ids:
        # Instance id 0 belongs to the background. Since the new_seg_mask is
        # initialized to 0's (background label), we do not have to explicitly
        # handle this case.
        if instance_id == 0:
            continue

        # Get the pixels corresponding to this instance_id.
        pixels = (seg_mask[:, :, 1] == instance_id)

        # The red_channel contains encodes the old_id. We adapt the code from
        # https://groups.csail.mit.edu/vision/datasets/ADE20K/code/loadAde20K.m
        # to extract the old_id (the integer label).
        red_channel_val = seg_mask[pixels][0, 0]
        old_id = red_channel_val // 10 * 256 + instance_id - 1

        # If we're not keeping this category in new dataset, discard it.
        if old_id not in old_id_to_new_id:
            continue

        new_id = old_id_to_new_id[old_id]

        # Create a bounding box to store information for object
        # detection tasks.
        [x, y, w, h] = cv2.boundingRect(pixels.astype(np.uint8))
        bbox = [x, y, w, h]

        # NOTE: We're not using the Polygon format for segmentations,
        # but instead RLE (https://en.wikipedia.org/wiki/Run-length_encoding).
        # Ensure that your framework supports this, you may need to
        # enable a flag of some kind in your config files or scripts.
        rle = cocomask.encode(np.asfortranarray(pixels.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')

        # Put all of the annotation metadata inside one dict.
        annotation = {
            'id': next_ann_id,
            'image_id': next_file_id,
            'segmentation': rle,
            'category_id': new_id,
            'iscrowd': 0,
            'area': int(np.sum(pixels)),
            'bbox': bbox,
        }

        # Add that annotation to our list of annotations, increment
        # annotation count.
        local_annotations.append(annotation)
        next_ann_id += 1


    # If we found no annotations, we found no instances of our categories
    # of interest. Don't add this image to our dataset.
    if local_annotations == []:
        return

    # Copy original image file to new location.
    dest_img_fp = os.path.join(img_dir, f'{next_file_id}.jpg')
    shutil.copy(img_fp, dest_img_fp)

    # Extract image metadata and save in COCO-compatible JSON format.
    (height, width, _) = seg_mask.shape
    image_info = create_coco_image_info(img_fp, next_file_id, width, height)
    conversion_metadata['images'].append(image_info)

    # Update remaining elements in the conversion_metadata
    conversion_metadata['annotations'].extend(local_annotations)
    conversion_metadata['next_ann_id'] = next_ann_id
    conversion_metadata['next_file_id'] += 1


def convert_to_coco(src_dir,
        dst_dir,
        required_keywords=[],
        reject_keywords=[],
        want_labels=['background',]):

    metadata = filter_ade20k_dataset(src_dir, required_keywords,
            reject_keywords)

    (label_to_new_id, new_id_to_label, \
            old_id_to_new_id) = get_new_label_mappings(src_dir, want_labels)

    # COCO format requires us to only provide non-background labels and to begin
    # the counts at index 1.
    labelmap_coco_format = [
            {"id": new_id, "name": label} for (new_id,
                label) in new_id_to_label.items() if new_id != 0
    ]

    (trn_dir, val_dir, trn_annot_fpath, val_annot_fpath) = create_coco_directories(dst_dir)

    partition2filepaths = {
        "training": (trn_dir, trn_annot_fpath),
        "validation": (val_dir, val_annot_fpath),
    }

    for partition, samples in metadata.items():
        print(f"Processing partition {partition}")

        img_dir, annot_fpath = partition2filepaths[partition]
        # Pass this between functions as needed to keep track of information
        # to go in the outputted JSON file.
        conversion_metadata = {
            'categories': labelmap_coco_format,
            'images': [],
            'annotations': [],
            'next_file_id': 0,
            'next_ann_id': 0,
        }

        for idx, (img_fp, seg_fp) in enumerate(samples):
            if (idx + 1) % 500 == 0:
                print(f"Processed [{idx+1}]/[{len(samples)}]")

            process_coco_sample(img_dir, conversion_metadata,
                    old_id_to_new_id, img_fp, seg_fp)


        # This is the COCO-formatted JSON annotations payload that we'll
        # save for this partition.
        coco_payload = {
            'categories': labelmap_coco_format,
            'images': conversion_metadata['images'],
            'annotations': conversion_metadata['annotations'],
        }

        print(f"Saving to {annot_fpath}")
        with open(annot_fpath, 'w') as wf:
            json.dump(coco_payload, wf)

        print(f"Completed {partition} partition")
        print(f"Kept a total of {len(coco_payload['images'])} of the {len(samples)} images")


def disp_seg(new_seg_mask, new_id_to_label, dpi=200):
    """
    Produces an image from the 1-channel segmentation with labels assigned
    according to the LABEL_NAMES from refinement.py.

    Args:
        :new_seg_mask: - 1-channel numpy array
        :new_id_to_label: - A mapping from integer label to text label.
        :dpi: - Dots Per Inch, higher value means larger image.

    """
    unique_vals = sorted(np.unique(new_seg_mask).tolist())
    text_labels = [new_id_to_label[v] for v in unique_vals]

    # Create a new label encoding, use only values 0 - len(unique_vals)
    new_seg = new_seg_mask * 0
    for idx, val in enumerate(unique_vals):
        new_seg[(new_seg_mask == val)] = idx

    fig = plt.figure(dpi=dpi)
    im = plt.imshow(new_seg)

    # Hide the grid
    plt.axis("off")

    # Get the colors of the values, according to thecolormap used by imshow.
    colors = [im.cmap(im.norm(value)) for value in np.unique(new_seg)]

    # Create a patch (proxy artist) for every color.
    patches = [mpatches.Patch(color=c, label=t[:15]) for (c, t) in zip(colors, text_labels)]

    # Put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
