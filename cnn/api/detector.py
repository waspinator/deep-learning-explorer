#!/usr/bin/python3

import os
import sys
import numpy as np

from keras.preprocessing.image import img_to_array

sys.path.insert(0, '../libraries')
from cnn.config import Config
import cnn.cnn
from PIL import Image

HOME_DIR = '/home/keras'
ROOT_DATA_DIR = os.path.join(HOME_DIR, "data")
WEIGHTS_DIR = os.path.join(ROOT_DATA_DIR, "weights")
MODEL_DIR = os.path.join(ROOT_DATA_DIR, "logs")


class MnistConfig(Config):
    NAME = "mnist_cnn"
    EPOCHS = 1
    EPOCH_STEPS = 100
    CLASSES = 10  # 10 digits
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28


class ObjectDetector(object):

    def __init__(self):

        self.config = MnistConfig()
        self.model = cnn.cnn.CNN(config=self.config, model_dir=MODEL_DIR)
        weights = cnn.cnn.find_last(self.config, MODEL_DIR)[1]
        self.model.keras_model.load_weights(weights)

        # https://github.com/keras-team/keras/issues/2397
        dummy_input = Image.new(
            'L', (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
        self.detect(dummy_input)

    def detect(self, image):

        prediction = self.model.predict(image)

        return int(prediction)
