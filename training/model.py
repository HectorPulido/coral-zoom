"""
"""

import tensorflow as tf
from tensorflow.keras import layers


def res_block(num_filters, inputs):
    """ """
    x = layers.SeparableConv2D(num_filters, 3, padding="same", activation="leaky_relu")(
        inputs
    )
    x = layers.SeparableConv2D(num_filters, 3, padding="same")(x)
    x = layers.Add()([inputs, x])
    return x


def upsample(inputs, num_filters, **kwargs):
    """ """
    initializer = tf.random_normal_initializer(0.0, 0.02)
    inputs = layers.Conv2DTranspose(
        num_filters,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        use_bias=False,
    )(inputs)

    inputs = layers.Conv2D(num_filters, 3, padding="same", **kwargs)(inputs)

    return inputs


def make_model(
    lr_shape, num_filters, num_of_residual_blocks_a, num_of_residual_blocks_b
):
    """ """
    # Flexible Inputs to input_layer
    input_layer = layers.Input(shape=lr_shape)
    # Scaling Pixel Values
    x = layers.Rescaling(scale=1.0 / 255)(input_layer)

    x = x_new = layers.Conv2D(num_filters, 3, padding="same")(x)

    # 16 residual blocks
    for _ in range(num_of_residual_blocks_a):
        x_new = res_block(num_filters, x_new)

    x_new = layers.Conv2D(num_filters, 3, padding="same")(x_new)
    x = layers.Add()([x, x_new])

    x = upsample(x, num_filters)
    x = x_new = upsample(x, num_filters)

    for _ in range(num_of_residual_blocks_b):
        x_new = res_block(num_filters, x_new)

    x_new = layers.Conv2D(num_filters, 3, padding="same")(x_new)
    x = layers.Add()([x, x_new])

    x = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)

    output_layer = layers.Rescaling(scale=255)(x)
    m = tf.keras.Model(input_layer, output_layer)
    return m
