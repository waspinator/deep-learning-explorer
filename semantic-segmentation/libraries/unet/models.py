#!/usr/bin/python3

import keras
import keras.layers as KL
import keras.models as KM

import unet.blocks as blocks


def unet(input_shape=(572, 572, 1), classes=2):
    """Original U-Net model
    """
    input_image = KL.Input(shape=input_shape)

    contracting_1, pooled_1 = blocks.contracting(input_image,   filters=64, block_name="block1")
    contracting_2, pooled_2 = blocks.contracting(pooled_1,      filters=128, block_name="block2")
    contracting_3, pooled_3 = blocks.contracting(pooled_2,      filters=256, block_name="block3")
    contracting_4, pooled_4 = blocks.contracting(pooled_3,      filters=512, block_name="block4")
    contracting_5, _ = blocks.contracting(pooled_4,             filters=1024, block_name="block5")

    dropout = KL.Dropout(rate=0.5)(contracting_5)

    expanding_1 = blocks.expanding(dropout,     merge_layer=contracting_4, filters=512, block_name="block6")
    expanding_2 = blocks.expanding(expanding_1, merge_layer=contracting_3, filters=256, block_name="block7")
    expanding_3 = blocks.expanding(expanding_2, merge_layer=contracting_2, filters=128, block_name="block8")
    expanding_4 = blocks.expanding(expanding_3, merge_layer=contracting_1, filters=64, block_name="block9")

    class_output = KL.Conv2D(classes, kernel_size=(1, 1), activation='softmax', name='class_output')(expanding_4)

    model = KM.Model(inputs=[input_image], outputs=[class_output])

    return model


def unet_vgg16(input_shape=(224, 224, 3), classes=1000):
    """U-NET model using VGG16
    """
    input_image = KL.Input(shape=input_shape)
    contracting_1, pooled_1 = blocks.contracting_vgg16(input_image, filters=64,  convolutions=2, block_name="block1")
    contracting_2, pooled_2 = blocks.contracting_vgg16(pooled_1, filters=128, convolutions=2, block_name="block2")
    contracting_3, pooled_3 = blocks.contracting_vgg16(pooled_2, filters=256, convolutions=3, block_name="block3")
    contracting_4, pooled_4 = blocks.contracting_vgg16(pooled_3, filters=512, convolutions=3, block_name="block4")
    contracting_5, _ = blocks.contracting_vgg16(pooled_4, filters=512, convolutions=3, block_name="block5")

    dropout = KL.Dropout(rate=0.5)(contracting_5)

    expanding_1 = blocks.expanding(dropout,     merge_layer=contracting_4, filters=512, block_name="block6")
    expanding_2 = blocks.expanding(expanding_1, merge_layer=contracting_3, filters=256, block_name="block7")
    expanding_3 = blocks.expanding(expanding_2, merge_layer=contracting_2, filters=128, block_name="block8")
    expanding_4 = blocks.expanding(expanding_3, merge_layer=contracting_1, filters=64, block_name="block9")

    class_output = KL.Conv2D(classes, kernel_size=(1, 1), activation='softmax', name='class_output')(expanding_4)

    model = KM.Model(inputs=[input_image], outputs=[class_output])

    return model


def unet_resnet50(input_shape, classes):
    pass


def unet_xception(input_shape, classes):
    pass
