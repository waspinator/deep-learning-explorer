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
import keras.models as KM
import keras.callbacks as KC
import keras.layers as KL
import keras.engine as KE
import keras.optimizers as KO

import cnn.models


class CNN(object):
    """Encapsulates CNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config, model_dir):
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(config=config)

    def build(self, config):
        model = cnn.models.CNN(
            input_shape=config.INPUT_SHAPE,
            classes=config.CLASSES
        )

        return model

    def compile(self):
        self.keras_model.compile(loss=keras.losses.categorical_crossentropy,
                                 optimizer=KO.Adadelta(),
                                 metrics=['accuracy'])

    def train(self, dataset_train, dataset_validate, epochs=None):

        if epochs is None:
            epochs = self.config.EPOCHS

        # create callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # compile model
        self.compile()

        # train model for selected number of epochs
        self.keras_model.fit(dataset_train.get('inputs'), dataset_train.get('labels'),
                             validation_data=(
                                 dataset_validate.get('inputs'), dataset_validate.get('labels')),
                             batch_size=self.config.BATCH_SIZE,
                             epochs=epochs,
                             verbose=1,
                             callbacks=callbacks
                             )

    def predict(self, pil_image):

        pil_image = pil_image.convert('L')

        pil_image = pil_image.resize(
            (self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH))

        np_image = np.asarray(pil_image)

        if KB.image_data_format() == 'channels_first':
            x = np_image.reshape(1, 1, 28, 28)
        else:
            x = np_image.reshape(1, 28, 28, 1)

        result = np.argmax(self.keras_model.predict(x, batch_size=1))

        return result

    def evaluate(self, dataset_test):
        score = self.keras_model.evaluate(
            dataset_test.get('inputs'),
            dataset_test.get('labels'),
            verbose=0)

        return score

    def set_log_dir(self, model_path=None):

        self.epoch = 0
        now = datetime.datetime.now()

        if model_path:

            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/model\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) - 1 + 1

        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        self.checkpoint_path = os.path.join(self.log_dir, "model_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")


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
    checkpoints = filter(lambda f: f.startswith("model"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        return dir_name, None
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return dir_name, checkpoint
