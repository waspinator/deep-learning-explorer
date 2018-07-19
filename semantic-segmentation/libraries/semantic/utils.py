#!/usr/bin/python3

import re
import os
import h5py
import datetime
import skimage
import scipy
import warnings
import random
import numpy as np
from PIL import Image
import pycocotools
import pycococreatortools.pycococreatortools as pycococreatortools

import keras.engine as KE


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
            epochs_drop = epochs / 10
            lr = initial_lr * lr_power ** int((1+epoch)/epochs_drop)

        return lr

    return lr_schedule


def load_weights(layers, filepath, by_name=False, exclude=None):
    """Modified version of the correspoding Keras function with
    support and the ability to exclude some layers from loading.
    exlude: list of layer names to excluce
    """

    if exclude:
        by_name = True

    f = h5py.File(filepath, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']

    # Exclude some layers
    if exclude:
        layers = filter(lambda l: l.name not in exclude, layers)

    if by_name:
        KE.topology.load_weights_from_hdf5_group_by_name(f, layers)
    else:
        KE.topology.load_weights_from_hdf5_group(f, layers)
    if hasattr(f, 'close'):
        f.close()


def set_trainable(layer_regex, keras_model, indent=0, verbose=1):
    """Sets model layers as trainable if their names match
    the given regular expression.
    """
    # Print message on the first call (but not on recursive calls)
    if verbose > 0 and keras_model is None:
        log("Selecting layers to train...")

    layers = keras_model.layers

    for layer in layers:
        # Is the layer a model?
        if layer.__class__.__name__ == 'Model':
            print("In model: ", layer.name)
            set_trainable(
                layer_regex, keras_model=layer, indent=indent + 4)
            continue

        if not layer.weights:
            continue
        # Is it trainable?
        trainable = bool(re.fullmatch(layer_regex, layer.name))
        # Update layer. If layer is a container, update inner layer.
        if layer.__class__.__name__ == 'TimeDistributed':
            layer.layer.trainable = trainable
        else:
            layer.trainable = trainable
        # Print trainble layer names
        if trainable and verbose > 0:
            log("{}{:20}   ({})".format(" " * indent, layer.name,
                                        layer.__class__.__name__))


def find_trainable_layer(layer):
    """If a layer is encapsulated by another layer, this function
    digs through the encapsulation and returns the layer that holds
    the weights.
    """
    if layer.__class__.__name__ == 'TimeDistributed':
        return find_trainable_layer(layer.layer)
    return layer


def get_trainable_layers(layers):
    """Returns a list of layers that have weights."""
    layers = []
    # Loop through all layers
    for l in layers:
        # If layer is a wrapper, find inner trainable layer
        l = find_trainable_layer(l)
        # Include layer if it has weights
        if l.get_weights():
            layers.append(l)
    return layers


def find_last(config, model_dir):
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Returns:
        dir_name: The directory where events and weights are saved
        checkpoint: the path to the last checkpoint file
    """
    # Get directory names. Each directory corresponds to a model
    dir_names = next(os.walk(model_dir))[1]
    key = config.NAME.lower()
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        return None, None
    # Pick last directory
    dir_name = os.path.join(model_dir, dir_names[-1])
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("fcn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        return dir_name, None
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return dir_name, checkpoint


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


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(h * scale), round(w * scale)),
            order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def resize_flat_mask(mask, scale, padding, crop=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding[0:2], mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = skimage.transform.resize(m, mini_shape, order=1, mode="constant")
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = skimage.transform.resize(m, (h, w), order=1, mode="constant")
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = skimage.transform.resize(
        mask, (y2 - y1, x2 - x1), order=1, mode="constant")
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


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
