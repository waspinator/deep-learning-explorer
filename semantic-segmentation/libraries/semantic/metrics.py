#!/usr/bin/python3

import keras.backend as KB
import tensorflow as tf


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = KB.int_shape(y_pred)[-1]
    y_pred = KB.reshape(y_pred, (-1, nb_classes))

    y_true = KB.one_hot(tf.to_int32(KB.flatten(y_true)),
                        nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return KB.sum(tf.to_float(legal_labels & KB.equal(KB.argmax(y_true, axis=-1), KB.argmax(y_pred, axis=-1)))) / KB.sum(tf.to_float(legal_labels))
