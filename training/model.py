"""
"""

import tensorflow as tf
from tensorflow.keras import layers


def res_block(num_filters, inputs):
    """ """
    x = layers.SeparableConv2D(num_filters, 3, padding="same", activation="relu")(
        inputs
    )
    x = layers.SeparableConv2D(num_filters, 3, padding="same")(x)
    x = layers.Add()([inputs, x])
    return x


def make_model(lr_shape, num_filters, num_of_residual_blocks_a=2, batch_size=None):
    """ """
    # Flexible Inputs to input_layer
    input_layer = layers.Input(shape=lr_shape, batch_size=batch_size)
    x = layers.Conv2D(num_filters, 3, padding="same", activation="relu")(input_layer)
    for _ in range(num_of_residual_blocks_a):
        x = res_block(num_filters, x)
    x = layers.SeparableConv2D(3, 3, padding="same", activation="sigmoid")(x)
    x = layers.Add()([input_layer, x])
    m = tf.keras.Model(input_layer, x)
    return m
