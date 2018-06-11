#!/usr/bin/python3

import os
import numpy as np
import keras.models as KM
import keras.layers as KL
from keras.applications.resnet50 import ResNet50
from semantic.blocks import conv_block, identity_block


def transfer_FCN_ResNet50(transfered_weights_path):
    input_shape = (224, 224, 3)
    img_input = KL.Input(shape=input_shape)
    bn_axis = 3

    x = KL.Conv2D(64, (7, 7), strides=(2, 2),
                  padding='same', name='conv1')(img_input)
    x = KL.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b')(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c')(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d')(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f')(x)

    x = conv_block(3, [512, 512, 2048], stage=5, block='a')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='b')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='c')(x)

    x = KL.Conv2D(1000, (1, 1), activation='linear', name='fc1000')(x)

    # Create model
    model = KM.Model(img_input, x)

    flattened_layers = model.layers
    index = {}
    for layer in flattened_layers:
        if layer.name:
            index[layer.name] = layer
    resnet50 = ResNet50()
    for layer in resnet50.layers:
        weights = layer.get_weights()
        if layer.name == 'fc1000':
            weights[0] = np.reshape(weights[0], (1, 1, 2048, 1000))
        if layer.name in index:
            index[layer.name].set_weights(weights)
    model.save_weights(transfered_weights_path)
    print('Successfully transformed!')
