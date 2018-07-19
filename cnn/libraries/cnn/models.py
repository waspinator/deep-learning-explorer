#!/usr/bin/python3

import keras.models as KM
import keras.layers as KL


def CNN(input_shape, classes):

    model = KM.Sequential()

    model.add(KL.Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
    model.add(KL.Conv2D(64, (3, 3), activation='relu'))
    model.add(KL.MaxPooling2D(pool_size=(2, 2)))
    model.add(KL.Dropout(0.25))
    model.add(KL.Flatten())
    model.add(KL.Dense(128, activation='relu'))
    model.add(KL.Dropout(0.5))
    model.add(KL.Dense(classes, activation='softmax'))

    return model
