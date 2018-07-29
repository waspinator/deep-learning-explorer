#!/usr/bin/python3
import numpy as np

import keras.backend as KB
import keras.preprocessing.image as KPI
import keras.utils as KU

import semantic.utils as sutils


class CocoGenerator(KPI.ImageDataGenerator):

    def __init__(self,
                 image_resample=True,
                 pixelwise_center=False,
                 pixel_mean=(0., 0., 0.),
                 pixelwise_std_normalization=False,
                 pixel_std=(1., 1., 1.),

                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 validation_split=0.0):

        self.image_resample = image_resample
        self.pixelwise_center = pixelwise_center
        self.pixel_mean = np.array(pixel_mean)
        self.pixelwise_std_normalization = pixelwise_std_normalization
        self.pixel_std = np.array(pixel_std)

        super().__init__()

    def standardize(self, x):
        if self.pixelwise_center:
            x -= self.pixel_mean
        if self.pixelwise_std_normalization:
            x /= self.pixel_std
        return super().standardize(x)

    def flow_from_dataset(self, dataset,
                          target_size=(256, 256), color_mode='rgb',
                          classes=None, class_mode='categorical',
                          batch_size=32, shuffle=True, seed=None,
                          interpolation='nearest'):

        return DatasetIterator(dataset, self,
                               target_size=target_size, color_mode=color_mode,
                               classes=classes, class_mode=class_mode,
                               data_format=self.data_format,
                               batch_size=batch_size, shuffle=shuffle, seed=seed,
                               interpolation=interpolation)


class DatasetIterator(KPI.Iterator):

    def __init__(self, dataset, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 data_format=None,
                 batch_size=32, shuffle=True, seed=None,
                 interpolation='nearest'):

        self.samples = len(dataset.image_ids)
        self.dataset = dataset
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')

        self.class_mode = class_mode
        self.interpolation = interpolation

        label_shape = list(self.image_shape)
        label_shape[self.image_data_generator.channel_axis - 1] = self.classes
        self.label_shape = tuple(label_shape)

        super().__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):

        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=KB.floatx())

        batch_y = np.zeros(
            (len(index_array),) + self.label_shape,
            dtype=np.int8)

        # build batch of image data
        for i, j in enumerate(index_array):

            max_dim = max(self.target_size)

            x = self.dataset.load_image(j)
            x = x.astype('float32')
            x, _, scale, padding, crop = sutils.resize_image(
                x, max_dim=max_dim)

            #params = self.image_data_generator.get_random_transform(x.shape)
            #x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            y = self.dataset.load_class_mask(j)
            y = sutils.resize_flat_mask(y, scale, padding, crop)
            y = KU.np_utils.to_categorical(
                y, self.classes).reshape(self.label_shape)
            batch_y[i] = y

        return batch_x, batch_y
