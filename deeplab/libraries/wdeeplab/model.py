#!/usr/bin/python3

import os
import random
import datetime
import re
import math
import logging
import h5py
from collections import OrderedDict
import multiprocessing
import numpy as np
import skimage.transform
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import keras.utils as KU


class DeepLabThreePlus():
    """Encapsulates the Deeplab v3+ model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Deeplabv3+ architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.

             This model is available for TensorFlow only,
            and can only be used with inputs following the TensorFlow
            data format `(width, height, channels)`.
            # Arguments
                weights: one of 'pascal_voc' (pre-trained on pascal voc)
                    or None (random initialization)
                input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                    to use as image input for the model.
                input_shape: shape of input image. format HxWxC
                    PASCAL VOC model was trained on (512,512,3) images
                classes: number of desired classes. If classes != 21,
                    last layer is initialized randomly
                OS: determines input_shape/feature_extractor_output ratio. One of {8,16}

            # Returns
                A Keras model instance.

            # Raises
                RuntimeError: If attempting to run this model with a
                    backend that does not support separable convolutions.
                ValueError: in case of invalid argument for `weights`
        """

        assert mode in ['training', 'inference']

        if K.backend() != 'tensorflow':
            raise RuntimeError('The Deeplabv3+ model is only available with '
                            'the TensorFlow backend.')

        if config.INPUT_SHAPE_OUTPUT_FEATURE_RATIO == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        input_image = KL.Input(shape=config.INPUT_SHAPE, name="input_image")

        x = KL.Conv2D(32, (3, 3), strides=(2, 2),
                name='entry_flow_conv1_1', use_bias=False, padding='same')(input_image)
        x = KL.BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = KL.Activation('relu')(x)

        x = self.conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = KL.BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = KL.Activation('relu')(x)

        x = self.xception_block(x, [128, 128, 128], 'entry_flow_block1',
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)
        x, skip1 = self.xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                skip_connection_type='conv', stride=2,
                                depth_activation=False, return_skip=True)

        x = self.xception_block(x, [728, 728, 728], 'entry_flow_block3',
                        skip_connection_type='conv', stride=entry_block3_stride,
                        depth_activation=False)
        for i in range(16):
            x = self.xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)

        x = self.xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                        depth_activation=False)
        x = self.xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                        depth_activation=True)
        # end of feature extractor

        # branching for Atrous Spatial Pyramid Pooling
        # simple 1x1
        b0 = KL.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = KL.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = KL.Activation('relu', name='aspp0_activation')(b0)

        # rate = 6 (12)
        b1 = self.SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = self.SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = self.SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # Image Feature branch
        out_shape = int(np.ceil(config.INPUT_SHAPE[0] / config.INPUT_SHAPE_OUTPUT_FEATURE_RATIO))
        b4 = KL.AveragePooling2D(pool_size=(out_shape, out_shape))(x)
        b4 = KL.Conv2D(256, (1, 1), padding='same',
                    use_bias=False, name='image_pooling')(b4)
        b4 = KL.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = KL.Activation('relu')(b4)
        b4 = BilinearUpsampling((out_shape, out_shape))(b4)

        # concatenate ASPP branches & project
        x = KL.Concatenate()([b4, b0, b1, b2, b3])
        x = KL.Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='concat_projection')(x)
        x = KL.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = KL.Activation('relu')(x)
        x = KL.Dropout(0.1)(x)

        # DeepLab v.3+ decoder

        # Feature projection
        # x4 (x2) block
        x = BilinearUpsampling(output_size=(int(np.ceil(config.INPUT_SHAPE[0] / 4)),
                                            int(np.ceil(config.INPUT_SHAPE[1] / 4))))(x)
        dec_skip1 = KL.Conv2D(48, (1, 1), padding='same',
                        use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = KL.BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = KL.Activation('relu')(dec_skip1)
        x = KL.Concatenate()([x, dec_skip1])
        x = self.SepConv_BN(x, 256, 'decoder_conv0',
                    depth_activation=True, epsilon=1e-5)
        x = self.SepConv_BN(x, 256, 'decoder_conv1',
                    depth_activation=True, epsilon=1e-5)

        # you can use it with arbitary number of classes
        if config.NUM_CLASSES == 21:
            last_layer_name = 'logits_semantic'
        else:
            last_layer_name = 'custom_logits_semantic'

        x = KL.Conv2D(config.NUM_CLASSES, (1, 1), padding='same', name=last_layer_name)(x)
        x = BilinearUpsampling(output_size=(config.INPUT_SHAPE[0], config.INPUT_SHAPE[1]))(x)

        inputs = [input_image]
        outputs = [x]

        model = KM.Model(inputs, outputs, name='deeplabv3+')

        return model


    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the ability to exclude some layers from loading.
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
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.h5"
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
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")




    def SepConv_BN(self, x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
        """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
            Implements right "same" padding for even kernel sizes
            Args:
                x: input tensor
                filters: num of filters in pointwise convolution
                prefix: prefix before name
                stride: stride at depthwise conv
                kernel_size: kernel size for depthwise convolution
                rate: atrous rate for depthwise convolution
                depth_activation: flag to use activation between depthwise & poinwise convs
                epsilon: epsilon to use in BN layer
        """

        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = KL.ZeroPadding2D((pad_beg, pad_end))(x)
            depth_padding = 'valid'

        if not depth_activation:
            x = KL.Activation('relu')(x)
        x = KL.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                            padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
        x = KL.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = KL.Activation('relu')(x)
        x = KL.Conv2D(filters, (1, 1), padding='same',
                use_bias=False, name=prefix + '_pointwise')(x)
        x = KL.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = KL.Activation('relu')(x)

        return x


    def conv2d_same(self, x, filters, prefix, stride=1, kernel_size=3, rate=1):
        """Implements right 'same' padding for even kernel sizes
            Without this there is a 1 pixel drift when stride = 2
            Args:
                x: input tensor
                filters: num of filters in pointwise convolution
                prefix: prefix before name
                stride: stride at depthwise conv
                kernel_size: kernel size for depthwise convolution
                rate: atrous rate for depthwise convolution
        """
        if stride == 1:
            return KL.Conv2D(filters,
                        (kernel_size, kernel_size),
                        strides=(stride, stride),
                        padding='same', use_bias=False,
                        dilation_rate=(rate, rate),
                        name=prefix)(x)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = KL.ZeroPadding2D((pad_beg, pad_end))(x)
            return KL.Conv2D(filters,
                        (kernel_size, kernel_size),
                        strides=(stride, stride),
                        padding='valid', use_bias=False,
                        dilation_rate=(rate, rate),
                        name=prefix)(x)


    def xception_block(self, inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
        """ Basic building block of modified Xception network
            Args:
                inputs: input tensor
                depth_list: number of filters in each SepConv layer. len(depth_list) == 3
                prefix: prefix before name
                skip_connection_type: one of {'conv','sum','none'}
                stride: stride at last depthwise conv
                rate: atrous rate for depthwise convolution
                depth_activation: flag to use activation between depthwise & pointwise convs
                return_skip: flag to return additional tensor after 2 SepConvs for decoder
                """
        residual = inputs
        for i in range(3):
            residual = self.SepConv_BN(residual,
                                depth_list[i],
                                prefix + '_separable_conv{}'.format(i + 1),
                                stride=stride if i == 2 else 1,
                                rate=rate,
                                depth_activation=depth_activation)
            if i == 1:
                skip = residual
        if skip_connection_type == 'conv':
            shortcut = self.conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
            shortcut = KL.BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
            outputs = KL.add([residual, shortcut])
        elif skip_connection_type == 'sum':
            outputs = KL.add([residual, inputs])
        elif skip_connection_type == 'none':
            outputs = residual
        if return_skip:
            return outputs, skip
        else:
            return outputs



class BilinearUpsampling(KE.Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = KU.conv_utils.normalize_data_format(data_format)
        self.input_spec = KE.InputSpec(ndim=4)
        if output_size:
            self.output_size = KU.conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = KU.conv_utils.normalize_tuple(upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)