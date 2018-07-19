#!/usr/bin/python3

import keras.backend as KB


class Config(object):

    BATCH_SIZE = 128
    CLASSES = 10
    EPOCHS = 12
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28

    def __init__(self):

        if KB.image_data_format() == 'channels_first':
            self.INPUT_SHAPE = (1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        else:
            self.INPUT_SHAPE = (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1)

    def display(self):
        """Display Configuration values."""
        print("\nConfiguration:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
