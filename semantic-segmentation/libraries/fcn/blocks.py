#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import keras.backend as KB
import keras.layers as KL
import keras.regularizers as KR


def identity_block(kernel_size, filters, stage, block, weight_decay=0., batch_momentum=0.99):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if KB.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                      kernel_regularizer=KR.l2(weight_decay))(input_tensor)
        x = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size),
                      padding='same', name=conv_name_base + '2b', kernel_regularizer=KR.l2(weight_decay))(x)
        x = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                      '2c', kernel_regularizer=KR.l2(weight_decay))(x)
        x = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu')(x)
        return x
    return f


def conv_block(kernel_size, filters, stage, block, weight_decay=0., strides=(2, 2), batch_momentum=0.99):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if KB.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', kernel_regularizer=KR.l2(weight_decay))(input_tensor)
        x = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', kernel_regularizer=KR.l2(weight_decay))(x)
        x = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                      '2c', kernel_regularizer=KR.l2(weight_decay))(x)
        x = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', kernel_regularizer=KR.l2(weight_decay))(input_tensor)
        shortcut = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu')(x)
        return x
    return f

# Atrous-Convolution version of residual blocks


def atrous_identity_block(kernel_size, filters, stage, block, weight_decay=0., atrous_rate=(2, 2), batch_momentum=0.99):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if KB.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                      kernel_regularizer=KR.l2(weight_decay))(input_tensor)
        x = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate,
                      padding='same', name=conv_name_base + '2b', kernel_regularizer=KR.l2(weight_decay))(x)
        x = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                      '2c', kernel_regularizer=KR.l2(weight_decay))(x)
        x = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu')(x)
        return x
    return f


def atrous_conv_block(kernel_size, filters, stage, block, weight_decay=0., strides=(1, 1), atrous_rate=(2, 2), batch_momentum=0.99):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if KB.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', kernel_regularizer=KR.l2(weight_decay))(input_tensor)
        x = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', dilation_rate=atrous_rate,
                      name=conv_name_base + '2b', kernel_regularizer=KR.l2(weight_decay))(x)
        x = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                      '2c', kernel_regularizer=KR.l2(weight_decay))(x)
        x = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', kernel_regularizer=KR.l2(weight_decay))(input_tensor)
        shortcut = KL.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu')(x)
        return x
    return f


def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = KB.image_data_format()
    if data_format == 'channels_first':
        original_shape = KB.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(
                np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(
                np.array([height_factor, width_factor]).astype('int32'))
        X = KB.permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = KB.permute_dimensions(X, [0, 3, 1, 2])
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape(
                (None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = KB.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(
                np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(
                np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape(
                (None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)
