#!/usr/bin/python3

import datetime
import numpy as np
from PIL import Image
import pycocotools
import pycococreatortools.pycococreatortools as pycococreatortools


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)


def create_learning_rate_schedule(epochs, lr_base, lr_power, mode='power_decay'):

    def lr_schedule(epoch):

        if mode is 'power_decay':
            lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)
        elif mode is 'exponential_decay':
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch+1)
        elif mode is 'adam':
            lr = 0.001
        elif mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.9 * epochs:
                lr = 0.0001
            elif epoch > 0.75 * epochs:
                lr = 0.001
            elif epoch > 0.5 * epochs:
                lr = 0.01
            else:
                lr = 0.1
        elif mode is 'step_decay':
            initial_lr = lr_base
            epochs_drop = int(epochs / 10)
            lr = initial_lr * lr_power ** int((1+epoch)/epochs_drop)

        return lr

    return lr_schedule


def zero_pad_array(a, target_width, target_height):

    original_width = np.shape(a)[1]
    original_height = np.shape(a)[0]

    top_pad = round((target_height - original_height)/2)
    bottom_pad = (target_height - original_height) - top_pad
    left_pad = round((target_width - original_width)/2)
    right_pad = (target_width - original_width) - left_pad

    if len(np.shape(a)) == 3:
        pad_width = ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0))
    else:
        pad_width = ((top_pad, bottom_pad), (left_pad, right_pad))

    return np.pad(a, pad_width=pad_width, mode='constant')


def resize_array(a, size):
    image = Image.fromarray(a)
    image.thumbnail((size, size))

    return np.asarray(image)


def result_to_coco(result, class_names, width, height, tolerance=2, INFO=None, LICENSES=None):
    """Encodes semantic segmentation result into COCO format
    """

    if INFO is None:
        INFO = {
            "description": "Semantic Segmentation Result",
            "url": "https://github.com/waspinator/deep-learning-explorer",
            "version": "0.1.0",
            "year": 2018,
            "contributor": "waspinator",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

    if LICENSES is None:
        LICENSES = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]

    IMAGES = [
        {
            "id": 1,
            "width": width,
            "height": height,
            "license": 1
        }
    ]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": [],
        "images": IMAGES,
        "annotations": []
    }

    for class_info in class_names:
        if class_info['name'] == 'BG':
            continue

        category = {
            "id": class_info['id'],
            "name": class_info['name'],
            "supercategory": ""
        }

        coco_output["categories"].append(category)

        mask = result == class_info['id']

        annotation = pycococreatortools.create_annotation_info(
            annotation_id=1,
            image_id=1,
            category_info={"id": class_info['id'], "is_crowd": True},
            binary_mask=mask,
            image_size=(width, height),
            tolerance=tolerance
        )

        if annotation is not None:
            coco_output['annotations'].append(annotation)

    return coco_output
