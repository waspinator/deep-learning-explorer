#!/usr/bin/python3

import keras
import keras.layers as KL


def contracting(input_layer, filters, kernel_size=(3, 3), padding='same',
                block_name=""):
    """Builds a contracting block for original U-NET model

    input_layer: the layer to initially convolve
    filters: the number of filters to create
    kernel_size: convolutional kernel size
    padding: padding for the output. original paper uses "valid"
    """

    conv_a = KL.Conv2D(filters, kernel_size, activation='relu', padding=padding,
                       name='{}_contracting_conv_a'.format(block_name))(input_layer)
    conv_b = KL.Conv2D(filters, kernel_size, activation='relu', padding=padding,
                       name='{}_contracting_conv_b'.format(block_name))(conv_a)
    pool = KL.MaxPooling2D(pool_size=(2, 2), padding=padding,
                           name='{}_contracting_pool'.format(block_name))(conv_b)

    return conv_b, pool


def expanding(input_layer, merge_layer, filters, kernel_size=(3, 3), padding='same',
              block_name=""):

    input_layer = KL.UpSampling2D(size=(2, 2))(input_layer)

    conv_up = KL.Conv2D(filters, kernel_size=(2, 2), activation='relu',
                        padding='same', name='{}_expanding_conv_up'.format(block_name))(input_layer)

    conv_up_height, conv_up_width = int(conv_up.shape[1]), int(conv_up.shape[2])
    merge_height, merge_width = int(merge_layer.shape[1]), int(merge_layer.shape[2])

    crop_top = (merge_height - conv_up_height) // 2
    crop_bottom = (merge_height - conv_up_height) - crop_top
    crop_left = (merge_width - conv_up_width) // 2
    crop_right = (merge_width - conv_up_width) - crop_left

    cropping = ((crop_top, crop_bottom), (crop_left, crop_right))
    merge_layer = KL.Cropping2D(cropping)(merge_layer)
    merged = KL.concatenate([merge_layer, conv_up])

    conv_a = KL.Conv2D(filters, kernel_size, activation='relu', padding=padding,
                       name='{}_expanding_conv_a'.format(block_name))(merged)
    conv_b = KL.Conv2D(filters, kernel_size, activation='relu', padding=padding,
                       name='{}_expanding_conv_b'.format(block_name))(conv_a)

    return conv_b


def contracting_vgg16(input_layer, filters, kernel_size=(3, 3), padding='same',
                      convolutions=2, block_name=""):
    """Builds a contracting block for a U-NET model based on VGG16

    convolutions: one of 2 or 3
    """

    conv_1 = KL.Conv2D(filters, kernel_size, activation='relu',
                       padding=padding, name='{}_conv1'.format(block_name))(input_layer)

    conv_2 = KL.Conv2D(filters, kernel_size, activation='relu',
                       padding=padding, name='{}_conv2'.format(block_name))(conv_1)

    if convolutions == 3:
        conv_3 = KL.Conv2D(filters, kernel_size, activation='relu',
                           padding=padding, name='{}_conv3'.format(block_name))(conv_2)

        pool = KL.MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(block_name))(conv_3)
        return conv_3, pool

    else:
        pool = KL.MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(block_name))(conv_2)
        return conv_2, pool
