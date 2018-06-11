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

import semantic.models
import semantic.metrics
import semantic.losses
import semantic.utils

import semantic.SegDataGenerator as SegDataGenerator


class FCN(object):
    """Encapsulates FCN model functionality.
    https://arxiv.org/abs/1605.06211
    https://github.com/shelhamer/fcn.berkeleyvision.org

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(config=config)

    def build(self, config):

        model = semantic.models.AtrousFCN_Resnet50_16s(
            weight_decay=config.WEIGHT_DECAY,
            input_shape=config.IMAGE_SHAPE,
            batch_momentum=config.BATCH_MOMENTUM,
            classes=config.NUM_CLASSES)

        return model

    def train(self, train_dataset, validation_dataset,
              learning_schedule, epochs, layers, augmentation=None):

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "head": r"(classify.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(classify.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(classify.*)",
            "5+": r"(res5.*)|(bn5.*)|(classify.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        self.set_trainable(layers)
        self.compile(learning_schedule)

        # Data generators

        # generator mostly unchanged from https://github.com/aurora95/Keras-FCN
        train_file_path = ''
        data_dir = ''
        data_suffix = ''
        label_dir = ''
        label_suffix = ''
        classes = ''
        target_size = (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)
        label_cval = 0
        batch_size = self.config.BATCH_SIZE
        loss_shape = None

        train_datagen = SegDataGenerator.SegDataGenerator(dataset=train_dataset,
                                                          zoom_range=[
                                                              0.5, 2.0],
                                                          zoom_maintain_shape=True,
                                                          crop_mode='random',
                                                          crop_size=target_size,
                                                          # pad_size=(505, 505),
                                                          rotation_range=0.,
                                                          shear_range=0,
                                                          horizontal_flip=True,
                                                          channel_shift_range=20.,
                                                          fill_mode='constant',
                                                          label_cval=label_cval)

        train_generator = train_datagen.flow_from_directory(
            file_path=train_file_path,
            data_dir=data_dir, data_suffix=data_suffix,
            label_dir=label_dir, label_suffix=label_suffix,
            classes=classes,
            target_size=target_size, color_mode='rgb',
            batch_size=batch_size, shuffle=True,
            loss_shape=loss_shape,
            ignore_label=0
            # save_to_dir='Images/'
        )

        val_datagen = SegDataGenerator.SegDataGenerator(dataset=validation_dataset,
                                                        zoom_range=[
                                                            0.5, 2.0],
                                                        zoom_maintain_shape=True,
                                                        crop_mode='random',
                                                        crop_size=target_size,
                                                        # pad_size=(505, 505),
                                                        rotation_range=0.,
                                                        shear_range=0,
                                                        horizontal_flip=True,
                                                        channel_shift_range=20.,
                                                        fill_mode='constant',
                                                        label_cval=label_cval)

        validation_generator = val_datagen.flow_from_directory(
            file_path=train_file_path,
            data_dir=data_dir, data_suffix=data_suffix,
            label_dir=label_dir, label_suffix=label_suffix,
            classes=classes,
            target_size=target_size, color_mode='rgb',
            batch_size=batch_size, shuffle=True,
            loss_shape=loss_shape,
            ignore_label=0
            # save_to_dir='Images/'
        )

        # Callbacks
        callbacks = [
            KC.TensorBoard(log_dir=self.log_dir,
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False),

            KC.ModelCheckpoint(self.checkpoint_path,
                               verbose=0,
                               save_weights_only=True),

            KC.LearningRateScheduler(learning_schedule)
        ]

        self.keras_model.fit_generator(
            generator=train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=multiprocessing.cpu_count(),
            use_multiprocessing=True,
        )

        self.epoch = max(self.epoch, epochs)

    def predict(self, pil_image):

        pil_image.thumbnail(
            (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM))

        image = KP.image.img_to_array(pil_image)
        image = semantic.utils.zero_pad_array(
            image, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)
        image = np.expand_dims(image, axis=0)
        image = KA.imagenet_utils.preprocess_input(image)
        result = self.keras_model.predict(image, batch_size=1)
        result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)

        return result

    def compile(self, schedule):
        loss = semantic.losses.softmax_sparse_crossentropy_ignoring_last_label
        optimizer = KO.SGD(lr=schedule(0),
                           momentum=self.config.LEARNING_MOMENTUM)
        metrics = [semantic.metrics.sparse_accuracy_ignoring_last_label]

        self.keras_model.compile(loss=loss,
                                 optimizer=optimizer,
                                 metrics=metrics)

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """

        if exclude:
            by_name = True

        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        layers = self.keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            KE.topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            KE.topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            semantic.utils.log("Selecting layers to train...")

        keras_model = keras_model or self.keras_model

        layers = keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
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
                semantic.utils.log("{}{:20}   ({})".format(" " * indent, layer.name,
                                                           layer.__class__.__name__))

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

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

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("fcn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint
