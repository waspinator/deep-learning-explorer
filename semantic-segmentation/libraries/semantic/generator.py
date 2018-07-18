#!/usr/bin/python3

import keras.backend as KB
import keras.preprocessing.image as KPI


class CocoGenerator(KPI.ImageDataGenerator):

    def __init__(self,
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

        super().__init__()

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

        self.samples = 0
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

        super().__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        pass
