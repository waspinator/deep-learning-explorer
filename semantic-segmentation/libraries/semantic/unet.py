#!/usr/bin/python3
import os
import random
import datetime
import re

import semantic.config


class UnetConfig(semantic.config.Config):
    """Configuration for training on the Shapes dataset.
    Derives from the base Config class and overrides values specific
    to the Shapes dataset.
    """
    NAME = "example_unet_config"

    IMAGES_PER_GPU = 32
    DATASET_SAMPLES = 250

    STEPS_PER_EPOCH = DATASET_SAMPLES / IMAGES_PER_GPU

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + square, circle, triangle

    IMAGE_MAX_DIM = 320
    STEPS_PER_EPOCH = 128
    VALIDATION_STEPS = int(STEPS_PER_EPOCH / 10)
    BATCH_MOMENTUM = 0.9
    LEARNING_MOMENTUM = 0.9
    LR_BASE_BASE = 0.0001
    LR_POWER = 0.9
    WEIGHT_DECAY = 0.0001/2


class UNET(object):
    """Encapsulates U-Net model functionality.
    https://arxiv.org/abs/1505.04597
    https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config, model_dir):
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(config=config)

    def build(self, config):
        pass

    def train(self, train_dataset, validation_dataset,
              learning_schedule, epochs, layers, augmentation=None):
        pass

    def compile(self, schedule):
        pass

    def predict(self, pil_image):
        pass

    def load_weights(self, filepath, by_name=False, exclude=None):
        pass

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
