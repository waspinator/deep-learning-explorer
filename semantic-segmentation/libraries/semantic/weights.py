#!/usr/bin/python3

import os
import numpy as np
import keras


def vgg16_imagenet_weights_path(cache_dir='../../data'):
    fname = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    origin = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

    keras.utils.get_file(fname, origin,
                         md5_hash='64373286793e3c8b2b4e3219cbf3544b',
                         cache_subdir='weights',
                         cache_dir=cache_dir)

    return os.path.join(cache_dir, 'weights', 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')


def resnet50_imagenet_weights_path(cache_dir='../../data'):
    fname = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    origin = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    keras.utils.get_file(fname, origin,
                         md5_hash='a7b3fe01876f51b976af0dea6bc144eb',
                         cache_subdir='weights',
                         cache_dir=cache_dir)

    return os.path.join(cache_dir, 'weights', 'resnet50_weights_tf_dim_ordering_tf_kernels.h5')


def fcn_resnet50_imagenet_weights_path(cache_dir='../../data'):
    return os.path.join(cache_dir, 'weights', 'fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5')
