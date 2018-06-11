#!/usr/bin/python3

import os
import sys
import numpy as np

from keras.preprocessing.image import img_to_array

sys.path.insert(0, '../libraries')
from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize
from mrcnn.model import log
import mcoco.coco as coco
import mextra.utils as extra_utils
from PIL import Image

HOME_DIR = '/home/keras'
DATA_DIR = os.path.join(HOME_DIR, "data/shapes")
MODEL_DIR = os.path.join(DATA_DIR, "logs")
CLASS_NAMES = ['BG', 'square', 'circle', 'triangle']
IMAGE_SIZE = 64

rpn_anchor_template = (1, 2, 4, 8, 16)  # anchor sizes in pixels
rpn_anchor_scales = tuple(i * (IMAGE_SIZE // 32) for i in rpn_anchor_template)


class ModelConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_MIN_DIM = IMAGE_SIZE
    RPN_ANCHOR_SCALES = rpn_anchor_scales


class ObjectDetector(object):

    def __init__(self):

        self.inference_config = ModelConfig()

        self.model = modellib.MaskRCNN(mode="inference",
                                       config=self.inference_config,
                                       model_dir=MODEL_DIR)

        self.model_path = self.model.find_last()[1]
        self.model.load_weights(self.model_path, by_name=True)

        # https://github.com/keras-team/keras/issues/2397
        dummy_input = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        self.detect(dummy_input)

    def detect(self, image, tolerance=2):
        """Detect objects in image

        inputs: PIL image, polygon fidelity tolerance
        """
        image = image.convert('RGB')
        image.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
        image = img_to_array(image)
        result = self.model.detect([image])[0]
        width = np.shape(image)[1]
        height = np.shape(image)[0]

        coco = extra_utils.result_to_coco(result, CLASS_NAMES,
                                          (width, height), tolerance)

        return coco
