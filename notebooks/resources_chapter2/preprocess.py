"""
Helper functions and utilities to preprocess the ADE20K dataset.

@author Osama Sakhi
"""

from collections import defaultdict
from shutil import copyfile
from tqdm import tqdm
import json
import os
import re

from pathlib import Path
from PIL import Image


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
            reject_matches = [kw in searchable_name for kw in required_keywords]
            if len(reject_keywords) and any(reject_matches):
                continue

            # Otherwise, record the image and segmentation filepaths into the
            # dataset metadata.
            img_filepath = str(filepath)
            seg_filepath =  filepath.replace(".jpg", "_seg.png")

            sample_metadata = (img_filepath, seg_filepath)
            metadata[partition].append(sample_metadata)

    return metadata
