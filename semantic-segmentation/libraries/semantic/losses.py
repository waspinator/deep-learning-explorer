#!/usr/bin/python3

import keras.backend as KB
import tensorflow as tf


def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    '''
    Softmax cross-entropy loss function for pascal voc segmentation
    and models which do not perform softmax.
    tensorlow only
    '''
    y_pred = KB.reshape(y_pred, (-1, KB.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = KB.one_hot(tf.to_int32(KB.flatten(y_true)),
                        KB.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -KB.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = KB.mean(cross_entropy)

    return cross_entropy_mean


def binary_crossentropy_with_logits(ground_truth, predictions):
    '''
    Softmax cross-entropy loss function for coco segmentation
    and models which expect but do not apply sigmoid on each entry
    tensorlow only
    '''

    return KB.mean(KB.binary_crossentropy(ground_truth, predictions, from_logits=True), axis=-1)
