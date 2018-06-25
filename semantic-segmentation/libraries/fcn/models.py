#!/usr/bin/python3

import os
import keras
import keras.layers as KL
import keras.models as KM
import keras.regularizers as KR
import keras.applications as KA
import fcn2.layers as SL
import fcn2.blocks as SB


def AtrousFCN_Resnet50_16s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = KL.Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = KL.Input(shape=input_shape)
        image_size = input_shape[0:2]

    batch_normalization_axis = 3

    x = KL.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                  name='conv1', kernel_regularizer=KR.l2(weight_decay))(img_input)
    x = KL.BatchNormalization(axis=batch_normalization_axis, name='bn_conv1',
                              momentum=batch_momentum)(x)
    x = KL.Activation('relu')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = SB.conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(
        1, 1), batch_momentum=batch_momentum)(x)
    x = SB.identity_block(3, [64, 64, 256], stage=2, block='b',
                          weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = SB.identity_block(3, [64, 64, 256], stage=2, block='c',
                          weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = SB.conv_block(3, [128, 128, 512], stage=3, block='a',
                      weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = SB.identity_block(3, [128, 128, 512], stage=3, block='b',
                          weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = SB.identity_block(3, [128, 128, 512], stage=3, block='c',
                          weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = SB.identity_block(3, [128, 128, 512], stage=3, block='d',
                          weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = SB.conv_block(3, [256, 256, 1024], stage=4, block='a',
                      weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = SB.identity_block(3, [256, 256, 1024], stage=4, block='b',
                          weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = SB.identity_block(3, [256, 256, 1024], stage=4, block='c',
                          weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = SB.identity_block(3, [256, 256, 1024], stage=4, block='d',
                          weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = SB.identity_block(3, [256, 256, 1024], stage=4, block='e',
                          weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = SB.identity_block(3, [256, 256, 1024], stage=4, block='f',
                          weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = SB.atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay, atrous_rate=(
        2, 2), batch_momentum=batch_momentum)(x)
    x = SB.atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(
        2, 2), batch_momentum=batch_momentum)(x)
    x = SB.atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(
        2, 2), batch_momentum=batch_momentum)(x)

    # classifying layer
    x = KL.Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear',
                  padding='same', strides=(1, 1), kernel_regularizer=KR.l2(weight_decay), name='classify')(x)
    x = SL.BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = KM.Model(img_input, x, name='AtrousFCN_Resnet50_16s')

    return model
