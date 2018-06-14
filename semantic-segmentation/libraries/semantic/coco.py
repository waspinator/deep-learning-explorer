"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (sudo pip install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil
import tempfile
from PIL import Image


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
import semantic.config
import semantic.dataset

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################


class CocoConfig(semantic.config.Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2


############################################################
#  Dataset
############################################################

class CocoDataset(semantic.dataset.Dataset):

    def __init__(self):
        self.coco = None
        self._class_weights = {}
        super().__init__()

    def prepare(self, class_map=None):
        super().prepare(class_map=None)

        dataset_class_weights = self.calculate_dataset_class_weights()
        for info in self.class_info:
            self._class_weights[info['id']] = dataset_class_weights[info['id']]

    @property
    def class_weight(self):
        return self._class_weights

    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        coco = COCO(
            "{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

        self.coco = coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones(
                            [image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        try:
            return self.image_info[image_id]['path']
        except:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def load_pil_mask(self, image_id, as_array=False):
        mask = self.load_mask(image_id)
        class_masks = mask[0]
        class_ids = np.unique(mask[1])

        combined_class_layers = []

        for class_id in class_ids:
            class_layer_ids = np.where(mask[1] == class_id)[0]
            class_layers = class_masks[:, :, class_layer_ids]

            class_masks_reveresed_order = []
            for layer in range(class_layers.shape[-1]):
                class_masks_reveresed_order.append(class_layers[..., layer])

            combined_class_layer = (np.logical_or.reduce(
                class_masks_reveresed_order)).astype(np.uint8)
            combined_class_layer[np.where(
                combined_class_layer == True)] = class_id

            combined_class_layers.append(combined_class_layer)

        current_layer = combined_class_layers[0]
        combined_classes = np.copy(current_layer)
        mask = (current_layer == 0)

        for i, layer in enumerate(combined_class_layers):
            if i == len(combined_class_layers)-1:
                break

            next_layer = combined_class_layers[i+1]
            layer_mask = (combined_classes == False)

            combined_classes[layer_mask] = next_layer[layer_mask]

        image = Image.fromarray(combined_classes)

        # png_file = tempfile.NamedTemporaryFile('wb')
        # image.save(png_file, 'PNG')

        if as_array:
            return np.asarray(image)

        return image

    def pil_to_coco(self, pil_image):
        pass

    def calculate_dataset_class_weights(self):
        dataset_class_weights = {}

        for image_id in self.image_ids:
            image_weights = self.calculate_image_class_weights(image_id)

            for class_id, weight in image_weights.items():
                if dataset_class_weights.get(class_id):
                    dataset_class_weights[class_id] += weight
                else:
                    dataset_class_weights[class_id] = weight

        for class_id in dataset_class_weights:
            dataset_class_weights[class_id] = \
                dataset_class_weights[class_id] / len(self.image_ids)

        return dataset_class_weights

    def calculate_image_class_weights(self, image_id):
        class_weights = {}
        total_object_weight = 0
        background_id = 0

        width = self.image_info[image_id]['width']
        height = self.image_info[image_id]['height']

        annotations = self.image_info[image_id]['annotations']

        for annotation in annotations:
            mask_rle = self.annToRLE(annotation, height, width)
            mask_area = maskUtils.area(mask_rle)
            weight = mask_area / (width * height)
            class_id = annotation['category_id']

            if class_weights.get(class_id):
                class_weights[class_id] += weight
            else:
                class_weights[class_id] = weight

            total_object_weight += weight

        class_weights[background_id] = 1 - total_object_weight

        return class_weights
