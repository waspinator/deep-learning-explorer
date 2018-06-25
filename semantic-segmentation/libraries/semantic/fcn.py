#!/usr/bin/python3

import os
import random
import datetime
import re
import math
import logging
import multiprocessing
import h5py
import numpy as np
import tensorflow as tf
import keras
import keras.backend as KB
import keras.callbacks as KC
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import keras.optimizers as KO
import keras.preprocessing as KP
import keras.applications as KA

from keras_fcn import FCN as FCN_VGG16


class FCN(object):
    """Encapsulates FCN model functionality.
    https://arxiv.org/abs/1605.06211
    https://github.com/shelhamer/fcn.berkeleyvision.org

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config, model_dir):

        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(config=config)

    def build(self, config):
        model = FCN_VGG16(
            input_shape=config.IMAGE_SHAPE,
            classes=config.NUM_CLASSES)

        return model

    def train(self, train_dataset, validation_dataset,
              learning_schedule, epochs, layers, augmentation=None):
        pass

    def predict(self, pil_image):
        pass

    def compile(self, schedule):
        self.keras_model.compile(optimizer='rmsprop',
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/fcn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/fcn\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "fcn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
