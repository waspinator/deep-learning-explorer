#!/usr/bin/python3
import keras.backend as K


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.dot(y_true, K.transpose(y_pred))
    union = K.dot(y_true, K.transpose(y_true))+K.dot(y_pred, K.transpose(y_pred))
    return (2. * intersection + smooth) / (union + smooth)


def dice_coef_loss(y_true, y_pred):
    return K.mean(1-dice_coef(y_true, y_pred), axis=-1)


def focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output),
                  axis=-1)
