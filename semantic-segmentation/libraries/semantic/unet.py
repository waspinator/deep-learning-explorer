#!/usr/bin/python3

import os
import sys
import random
import datetime
import re
import multiprocessing

import numpy as np
import keras
import keras_contrib

import semantic.config
import semantic.utils
import semantic.generator
import semantic.losses

sys.path.insert(0, '../libraries')
import unet.models


class Config(semantic.config.Config):
    """Configuration for training on the Shapes dataset.
    Derives from the base Config class and overrides values specific
    to the Shapes dataset.
    """
    NAME = "unet"

    # Number of classes (including background)
    CLASSES = 1 + 1  # ex: background + foreground

    IMAGE_HEIGHT = 572
    IMAGE_CHANNELS = 1  # ex: red + green + blue

    BATCH_SIZE = 1

    def __init__(self):
        super().__init__()


class Unet(object):
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
        model = unet.models.unet(config.IMAGE_SHAPE, config.CLASSES)
        return model

    def compile(self, learning_rate=0.001, momentum=0.9):

        #optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum)
        optimizer = keras.optimizers.Adam(1e-5)
        #loss = keras_contrib.losses.jaccard_distance
        #loss = keras.losses.categorical_crossentropy
        loss = semantic.losses.dice_coef_loss
        metrics = [keras.metrics.categorical_accuracy]

        self.keras_model.compile(optimizer, loss, metrics)

    def train(self, dataset_train, dataset_validate,
              epochs, layers, learning_rate, augmentation=None):

        # set trainable layers
        layer_regex = {
            "head": r"(class_output)",
            "expanding": r"(\S*expanding\S*)|(class_output)",
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        semantic.utils.set_trainable(layers, self.keras_model)

        # create data generators for training and validation datasets
        data_generator = semantic.generator.CocoGenerator(
                pixel_mean=self.config.PIXEL_MEAN,
                pixel_std=self.config.PIXEL_STANDARD_DEVIATION,
                pixelwise_center=True,
                pixelwise_std_normalization=True
        )

        generator_train = data_generator.flow_from_dataset(
            dataset_train,
            target_size=(self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            classes=self.config.CLASSES
        )

        generator_validate = data_generator.flow_from_dataset(
            dataset_validate,
            target_size=(self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            classes=self.config.CLASSES
        )

        # create callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # compile model
        self.compile(learning_rate)

        # train model for selected number of epochs
        self.keras_model.fit_generator(
            generator_train,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=generator_validate,
            validation_steps=self.config.VALIDATION_STEPS,
            workers=multiprocessing.cpu_count(),
            use_multiprocessing=False,
        )

        self.epoch = max(self.epoch, epochs)

    def predict(self, pil_image):
        pil_image.thumbnail(
            (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))

        image = keras.preprocessing.image.img_to_array(pil_image)

        pixel_mean = np.array(self.config.PIXEL_MEAN)
        pixel_std = np.array(self.config.PIXEL_STANDARD_DEVIATION)

        image -= pixel_mean
        image /= pixel_std

        image = semantic.utils.zero_pad_array(
            image, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH)
        image = np.expand_dims(image, axis=0)
        result = self.keras_model.predict(image, batch_size=1)
        result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)

        return result

    def evaluate(self):
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
            # /path/to/logs/coco20171029T2315/unet_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/unet\_\w+(\d{4})\.h5"
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
        self.checkpoint_path = os.path.join(self.log_dir, "unet_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
