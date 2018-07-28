"""
Semantic Segmentation
Base Configurations class.
"""

import math
import numpy as np
import sys


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.


class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # Number of classes (including background)
    CLASSES = 1 + 1  # background + foreground

    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 0

    IMAGE_CHANNELS = 3  # red, green, blue

    TRAIN_DATASET_SIZE = 1000
    BATCH_SIZE = 10
    STEPS_PER_EPOCH = 0
    VALIDATION_STEPS = 0

    def __init__(self):
        """Set values of computed attributes."""

        if self.IMAGE_WIDTH == 0:
            self.IMAGE_WIDTH = self.IMAGE_HEIGHT

        # Input image size
        self.IMAGE_SHAPE = (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS)

        if self.STEPS_PER_EPOCH == 0:
            self.STEPS_PER_EPOCH = self.TRAIN_DATASET_SIZE // self.BATCH_SIZE

        #h, w = self.IMAGE_SHAPE[:2]
        # if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        #    raise Exception("Image size must be dividable by 2 at least 6 times "
        #                    "to avoid fractions when downscaling and upscaling."
        #                    "For example, use 256, 320, 384, 448, 512, ... etc. ")

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
