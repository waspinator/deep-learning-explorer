#!/usr/bin/python

import datetime
import re
import os

import keras.utils as KU

PASCAL_VOC_TRAINED_WEIGHTS = 'https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.0/deeplabv3_weights_tf_dim_ordering_tf_kernels.h5'

def download_trained_weights(weights_path):
    """Downloads Pascal VOC trained weights from Tensorflow via Bonlime.
    Returns path to weights file.
    """
    
    weights_path = KU.data_utils.get_file('deeplabv3_weights_tf_dim_ordering_tf_kernels.h5',
                            PASCAL_VOC_TRAINED_WEIGHTS,
                            cache_dir=weights_path,
                            cache_subdir='weights',
                            md5_hash='60d1a8ba0964a97dcf272cc022567b4a')
    return weights_path