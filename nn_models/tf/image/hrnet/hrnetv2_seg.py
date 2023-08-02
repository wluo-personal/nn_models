"""
reference:
https://github.com/niecongchong/HRNet-keras-semantic-segmentation/blob/master/model/seg_hrnet.py
https://github.com/HRNet/HRNet-Semantic-Segmentation
"""

import tensorflow as tf

N_FILTERS_STEM_NET = 64

def make_a_keras_model(input_shape, func, **kwargs):
    """
    helper function to validate the input output shape
    :param input_shape: (256, 1800, 3) tuple of integers
    :param func:
    :param kwargs: the kwa
    :return:
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    outputs = func(inputs, **kwargs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def conv_block(inputs, out_filters, kernel_size=3, strides=(1, 1), bool_batchnorm=False, bool_activation=False):
    """
    basic block which contains Conv + BatchNorm(Optional) + Activation(Optional)
    :param inputs:
    :param out_filters:
    :param kernel_size:
    :param strides:
    :param bool_batchnorm:
    :param bool_activation:
    :return:
    """
    x = tf.keras.layers.Conv2D(
        out_filters,
        kernel_size,
        padding='same',
        strides=strides,
        use_bias=False, # on large dataset, no need to enable bias
        kernel_initializer='he_normal')(inputs)
    if bool_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
    if bool_activation:
        x = tf.keras.layers.Activation('relu')(x)
    return x

def basic_block(inputs, out_filters: int):
    """ basic block with skip connection
        the shape of inputs[-1] should be equal to out_filters, otherwise there will be exceptions

    This will be used to construct the branches.

    inputs.shape == outputs.shape
    """
    strides = (1, 1)
    kernel_size = 3
    x1 = conv_block(
        inputs=inputs,
        out_filters=out_filters,
        kernel_size=kernel_size,
        strides=strides,
        bool_batchnorm=True,
        bool_activation=True)

    # no activation
    x2 = conv_block(
        inputs=x1,
        out_filters=out_filters,
        kernel_size=kernel_size,
        strides=strides,
        bool_batchnorm=True,
        bool_activation=False)

    x = tf.keras.layers.add([x2, inputs])
    outputs = tf.keras.layers.Activation('relu')(x)
    return outputs

def bottleneck_block(inputs, out_filters,  with_conv_shortcut=False):
    """ This is majorly for the immediate layers after input layer
    the inputs.shape == outputs.shape
    """
    # This is used to handle the connection in stem_net method
    expansion = 4
    de_filters = int(out_filters / expansion)
    strides = (1, 1)
    
    x1 = conv_block(
        inputs=inputs,
        out_filters=de_filters,
        kernel_size=1,
        strides=strides,
        bool_batchnorm=True,
        bool_activation=True)

    x2 = conv_block(
        inputs=x1,
        out_filters=de_filters,
        kernel_size=3,
        strides=strides,
        bool_batchnorm=True,
        bool_activation=True)

    x3 = conv_block(
        inputs=x2,
        out_filters=out_filters,
        kernel_size=1,
        strides=strides,
        bool_batchnorm=True,
        bool_activation=False)

    if with_conv_shortcut:
        residual = conv_block(
        inputs=inputs,
        out_filters=out_filters,
        kernel_size=1,
        strides=strides,
        bool_batchnorm=True,
        bool_activation=False)
    else:
        residual = inputs
    x4 = tf.keras.layers.add([x3, residual])
    outputs = tf.keras.layers.Activation('relu')(x4)

    return outputs

def stem_net(inputs, out_filters=N_FILTERS_STEM_NET):
    """
    input_shape: (x1, x2, x3)
    output_shape: (x1//2, x2//2, x3 * 4)
    """
    # why reduce the size here and then upsample eventually? Why not to keep the original size by using
    # strides = (1,1)?
    # Guess: this is to reduce the cost of memory


    x1 = conv_block(
        inputs=inputs,
        out_filters=out_filters,
        kernel_size=3,
        strides=(2, 2),
        bool_batchnorm=True,
        bool_activation=True)

    # The # of bottleneck filters should be 4* out_filters
    x2 = bottleneck_block(x1, 4*out_filters, with_conv_shortcut=True)
    x3 = bottleneck_block(x2, 4*out_filters, with_conv_shortcut=False)
    x4 = bottleneck_block(x3, 4*out_filters, with_conv_shortcut=False)
    outputs = bottleneck_block(x4, 4*out_filters, with_conv_shortcut=False)

    return outputs




