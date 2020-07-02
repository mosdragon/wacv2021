"""
Helper functions and utilities to preprocess the ADE20K dataset.

@author Osama Sakhi
"""

from collections import defaultdict
import shutil
import json
import os
import re

from pathlib import Path
from PIL import Image

import numpy as np
import scipy.io
from imageio import imread, imsave

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def create_new_seg_mask(seg_mask, old_id_to_new_id, new_id_to_label):
    """
    """

    # Create a new one-channel mask
    h, w, _ = seg_mask.shape
    new_seg_mask = np.zeros((h, w)).astype(np.uint8)

    # The sed_mask's green channel stores the instance id.
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


def create_pascal_directories(dst_dir):
    # Create the same directory structure as PASCAL VOC 2012.
    VOC_DIR = os.path.join(dst_dir, 'VOCdevkit', 'VOC2012')

    # Create the destination image and segmentation directory if it
    # doesn't exist.
    DST_IMG_DIR = os.path.join(VOC_DIR, 'JPEGImages')
    DST_ANNOT_DIR = os.path.join(VOC_DIR, 'SegmentationClass')
    DST_IMGSET_DIR = os.path.join(VOC_DIR, 'ImageSets')
    DST_IMGSET_SEG_DIR = os.path.join(DST_IMGSET_DIR, 'Segmentation')

    Path(DST_IMG_DIR).mkdir(parents=True, exist_ok=True)
    Path(DST_ANNOT_DIR).mkdir(parents=True, exist_ok=True)
    Path(DST_IMGSET_DIR).mkdir(parents=True, exist_ok=True)

    # Create the other directories that PASCAL uses, even if we
    # leave them empty.
    Path(os.path.join(VOC_DIR, 'Annotations')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(VOC_DIR, 'SegmentationObject')).mkdir(parents=True, exist_ok=True)

    # These directories are all under the ImageSets Directory
    Path(os.path.join(DST_IMGSET_DIR, 'Action')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(DST_IMGSET_DIR, 'Layout')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(DST_IMGSET_DIR, 'Main')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(DST_IMGSET_DIR, 'Segmentation')).mkdir(parents=True, exist_ok=True)

    return (DST_IMG_DIR, DST_ANNOT_DIR, DST_IMGSET_SEG_DIR)


def convert_to_pascal(src_dir, dst_dir, required_keywords=[],
        reject_keywords=[],
        want_labels=['background',]):

    metadata = filter_ade20k_dataset(src_dir, required_keywords,
            reject_keywords)

    (label_to_new_id, new_id_to_label, \
        old_id_to_new_id) = get_new_label_mappings(src_dir, want_labels)

    (img_dir, annot_dir, imgset_dir) = create_pascal_directories(dst_dir)

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
            partition_fp = os.path.join(imgset_dir, "train.txt")
        else:
            partition_fp = os.path.join(imgset_dir, "val.txt")

        with open(partition_fp, 'w') as wf:
            for sample_name in kept_samples:
                wf.write(sample_name + "\n")

        print(f"Partition {partition} has {len(kept_samples)}.")


    # Store the new_id_to_label mapping in the destination directory.
    labelmap_fp = os.path.join(dst_dir, 'new_id_to_label.json')
    with open(labelmap_fp, 'w') as wf:
        json.dump(new_id_to_label, wf, indent=4)

    print(f"ADE20K dataset has been converted and stored in {dst_dir}")
    print(f"Label mapping has been stored in {labelmap_fp}")


def get_new_id_to_label(dst_dir):
    labelmap_fp = os.path.join(dst_dir, 'new_id_to_label.json')
    with open(labelmap_fp, 'r') as rf:
        labelmap_tmp = json.load(rf)

    new_id_to_label = {int(key): val for (key, val) in labelmap_tmp.items()}
    return new_id_to_label


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
