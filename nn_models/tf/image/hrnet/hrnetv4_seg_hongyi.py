"""
This is based on hrnetv2_seg.py for hongyi

v3 for the vin head is using the transformer!
"""

"""
reference:
https://github.com/niecongchong/HRNet-keras-semantic-segmentation/blob/master/model/seg_hrnet.py
https://github.com/HRNet/HRNet-Semantic-Segmentation
"""

import tensorflow as tf
import string

N_FILTERS_STEM_NET = 64
N_BOTTLENECK_LAYERS_IN_STEM = 3  # any non-negative integer is valid

# The main branch number of filters. The sub-branch will be this * 2
BASE_BRANCH_FILTERS = 32

# number of block in a branch
N_BLOCKS_PER_BRANCH = 4  # integer greater than 0. Orignal value is 4

# whether to use Con2Dtranspose to do upsampling
BOOL_UPSAMPLE_TRANSPOSE = False


class ModelHyperParams:
    @staticmethod
    def set_n_filters_STEM_NET(value: int = 64):
        global N_FILTERS_STEM_NET
        N_FILTERS_STEM_NET = value

    @staticmethod
    def set_n_booleneck_layers_in_STEM(value: int = 3):
        global N_BOTTLENECK_LAYERS_IN_STEM
        N_BOTTLENECK_LAYERS_IN_STEM = value

    @staticmethod
    def set_base_branch_filters(value: int = 32):
        global BASE_BRANCH_FILTERS
        BASE_BRANCH_FILTERS = value

    @staticmethod
    def set_n_blocks_per_branch(value: int = 32):
        global N_BLOCKS_PER_BRANCH
        N_BLOCKS_PER_BRANCH = value

    @staticmethod
    def set_bool_upsamle_transpose(value: bool = False):
        global BOOL_UPSAMPLE_TRANSPOSE
        BOOL_UPSAMPLE_TRANSPOSE = value


# TODO
# 1. try to change upsampling to con2dtranspose by enbling the bool to True

def make_a_keras_model(input_shape, func, **kwargs):
    """
    helper function to validate the input output shape
    :param input_shape: (256, 1800, 3) tuple of integers
    :param func:
    :param kwargs: the kwa
    :return:
    """
    if isinstance(input_shape[0], tuple):
        inputs = []
        for each_input_shape in input_shape:
            inputs.append(tf.keras.layers.Input(shape=each_input_shape))
    else:
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
        use_bias=False,  # on large dataset, no need to enable bias
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


def bottleneck_block(inputs, out_filters, with_conv_shortcut=False):
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


def stem_net(inputs, out_filters=N_FILTERS_STEM_NET, n_bottleneck_layers=N_BOTTLENECK_LAYERS_IN_STEM):
    """
    input_shape: (x1, x2, x3)
    output_shape: (x1//2, x2//2, N_FILTERS_STEM_NET * 4)
    """
    # why reduce the size here and then upsample eventually? Why not to keep the original size by using
    # strides = (1,1)?
    # Guess: this is to reduce the cost of memory

    x = conv_block(
        inputs=inputs,
        out_filters=out_filters,
        kernel_size=3,
        strides=(2, 2),
        bool_batchnorm=True,
        bool_activation=True)

    # The # of bottleneck filters should be 4* out_filters
    x = bottleneck_block(x, 4 * out_filters, with_conv_shortcut=True)
    for _ in range(n_bottleneck_layers):
        x = bottleneck_block(x, 4 * out_filters, with_conv_shortcut=False)
    return x


def _construct_transition_layer(x: tuple, out_filters_list: tuple or list):
    """ Get transition layers which is responsible for split a new branch. Only the last one is split from the last input

    x: list of inputs, size n
    out_filters_list: tuple or list of integers, size n + 1
    """
    assert len(x) == len(out_filters_list) - 1
    outputs = []
    # For those original patches
    for inputs, n_filters in zip(x, out_filters_list[:-1]):
        x_ = conv_block(inputs=inputs, out_filters=n_filters, strides=(1, 1),
                        kernel_size=3, bool_batchnorm=True, bool_activation=True)
        outputs.append(x_)

    # split from the last path. The size was cut by 2
    x_ = conv_block(inputs=x[-1], out_filters=out_filters_list[-1], strides=(2, 2),
                    kernel_size=3, bool_batchnorm=True, bool_activation=True)
    outputs.append(x_)
    return outputs


def construct_transition_layer(x: tuple, base_branch_filters=BASE_BRANCH_FILTERS):
    """ refer to _construct_transition_layer
    construct a transition layer by specify the list of x. A new branch will be split by end of x.
    >>> construct_transition_layer([tf.keras.layers.Input(shape=(300, 300, 3))], base_branch_filters=32)
    """

    out_filters_list = [base_branch_filters * (2 ** i) for i in range(0, len(x) + 1)]
    return _construct_transition_layer(x, out_filters_list)


def upsample(x, filters, size=(2, 2), use_transpose=False):
    if use_transpose:
        x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=size, strides=size, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        return x
    else:
        x = conv_block(x, out_filters=filters, kernel_size=1, strides=(1, 1), bool_batchnorm=True,
                       bool_activation=False)
        return tf.keras.layers.UpSampling2D(size=size)(x)


def fuse_to_single_output(x: tuple, loc_calibrate: int, base_layer_filters=BASE_BRANCH_FILTERS,
                          bool_upsample_transpose=BOOL_UPSAMPLE_TRANSPOSE):
    """
    This method use to interact between different branches by selecting a targe branch and downscale/upscale other
    branches and then adding them together
    The size of x[i-1] is 1 times larger than x[i]

    :param x: tuple of inputs. Should be >= 2
    :param loc_calibrate: integer (zero index), where input to be as target to calibrate
    :param base_layer_filters: the x[0] # of filter
    :param bool_upsample_transpose: bool, whether to
        True:  use Con2Dtranspose to do upsampling
        False: use Upsample to do upsampling which is simple and not required to much training

    :return:
    """

    add_list = []

    # down scale
    for idx in range(loc_calibrate):
        inputs = x[idx]
        distance = abs(idx - loc_calibrate)
        x_ = inputs
        filters = base_layer_filters * int(2 ** idx)
        size = (2, 2)
        # each time only scale down by 2
        for i in range(distance):
            # the last down scale no need to do activation
            if i == distance - 1:
                bool_activate = False
            else:
                bool_activate = True
            filters = int(2 * filters)
            x_ = conv_block(inputs=x_, out_filters=filters, kernel_size=3, strides=size,
                            bool_batchnorm=True, bool_activation=bool_activate)

        add_list.append(x_)

    # loc_calibrate
    x_ = x[loc_calibrate]
    filters = base_layer_filters * int(2 ** loc_calibrate)
    x_ = conv_block(inputs=x_, out_filters=filters, kernel_size=1, strides=(1, 1),
                    bool_batchnorm=True, bool_activation=False)
    add_list.append(x_)

    # up sample
    for idx in range(loc_calibrate + 1, len(x)):
        inputs = x[idx]
        distance = abs(idx - loc_calibrate)
        x_ = inputs
        filters = base_layer_filters * int(2 ** loc_calibrate)
        # simple scale
        if not bool_upsample_transpose:
            nsize = int(2 ** distance)
            size = (nsize, nsize)
            x_ = upsample(x_, filters, size, use_transpose=bool_upsample_transpose)
        else:
            size = (2, 2)
            filters = base_layer_filters * int(2 ** idx)
            # each time only scale up by 2
            for i in range(distance):
                filters = int(filters / 2)
                x_ = upsample(x_, filters, size, use_transpose=bool_upsample_transpose)
                if i != distance - 1:  # need to activate
                    x_ = tf.keras.layers.Activation('relu')(x_)
        add_list.append(x_)
    x = tf.keras.layers.add(add_list)
    return x


def construct_fuse_layers(x: tuple, base_layer_filters=BASE_BRANCH_FILTERS,
                          bool_upsample_transpose=BOOL_UPSAMPLE_TRANSPOSE):
    """
    construct a fuse layer to merge all branches together
    """
    outputs = []
    for loc_calibrate in range(len(x)):
        out = fuse_to_single_output(
            x,
            loc_calibrate=loc_calibrate,
            base_layer_filters=base_layer_filters,
            bool_upsample_transpose=bool_upsample_transpose)
        outputs.append(out)
    return outputs


def make_single_branch(x: tf.keras.layers.Layer, out_filters=32, n_blocks=N_BLOCKS_PER_BRANCH):
    for _ in range(n_blocks):
        x = basic_block(x, out_filters=out_filters)
    return x


def make_layer_branches(x: tuple, base_filters=BASE_BRANCH_FILTERS, n_blocks=N_BLOCKS_PER_BRANCH):
    outs = []
    for idx, inputs in enumerate(x):
        n_filters = base_filters * (2 ** idx)
        out = make_single_branch(x=inputs, out_filters=n_filters, n_blocks=n_blocks)
        outs.append(out)
    return outs


def final_segmentation_layer(x, n_class, base_filters=BASE_BRANCH_FILTERS,
                             bool_upsample_transpose=BOOL_UPSAMPLE_TRANSPOSE,
                             name="segment_prob"):
    # since originally it is scaled down by 2,2. Upscale by 2*2 to original image size
    size = (2, 2)
    if bool_upsample_transpose:
        x = tf.keras.layers.Conv2DTranspose(
            base_filters, kernel_size=size, strides=size, padding="same", use_bias=False)(x)
    else:
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(n_class, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Softmax(axis=-1, name=name)(x)
    return x

def final_vin_layer(x, name="vin_prob"):
    N_VIN = 17
    OUTPUT_CHARS = 36
    # concats = [tf.keras.layers.GlobalAveragePooling2D()(x),
    #            tf.keras.layers.GlobalMaxPooling2D()(x)]
    x = tf.keras.layers.Conv2D(
        filters=1588,
        kernel_size=(1,1),
        strides=(1, 1),
        padding="valid",
        use_bias=False,
        kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # concats = []
    # concats.append(tf.keras.layers.GlobalAveragePooling2D()(x))
    # concats.append(tf.keras.layers.GlobalMaxPooling2D()(x))
    # x = tf.keras.layers.Concatenate()(concats)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    concats = []
    for _ in range(N_VIN):
        x_ = tf.keras.layers.Dense(units=OUTPUT_CHARS)(x)
        x_ = tf.keras.layers.Softmax(axis=-1)(x_)
        concats.append(x_)

    # x = tf.keras.layers.(name=name)(concats)
    x = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1), name=name)(concats)
    return x



def seg_prob_to_category(seg_prob, name="segment_category"):
    return tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis=-1), name=name)(seg_prob)

def vin_prob_to_string(vin_prob, name="vin_category"):
    vin_category = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis=-1), name=name)(vin_prob)
    return vin_category


def seg_hrnet(image_shape=(128, 1024, 3), n_class=20):
    """

    :param image_shape:
    :param n_class:
    :return:
    the output is a dictionary
    segment: shape is (H * W * n_class)
    vin: shape is (17 * 36)
    """
    name_segment_prob = "segment_prob"
    name_vin_prob = "vin_prob"
    name_vin_cat = "vin_category"
    name_segment_cat = "segment_category"

    inputs = tf.keras.layers.Input(shape=image_shape)
    # step 1: interact
    x = stem_net(inputs)
    # convert x to list to be compatible with below
    x = [x]

    # split branch -> grow branch -> fuse branch: 3 stages
    n_splits = 3
    for i in range(n_splits):  # 1->2 ->3 -> 4 branch
        x = construct_transition_layer(x)
        x = make_layer_branches(x)
        if i == n_splits - 1:  # last layer not do a full fuse
            x = fuse_to_single_output(x, loc_calibrate=0)
        else:
            x = construct_fuse_layers(x)
    # construct output layer
    seg_output_prob = final_segmentation_layer(x, n_class=n_class, name=name_segment_prob)
    seg_category = seg_prob_to_category(seg_prob=seg_output_prob, name=name_segment_cat)
    vin_output_prob = final_vin_layer(x, name=name_vin_prob)
    vin_string = vin_prob_to_string(vin_output_prob, name=name_vin_cat)

    # to connect outputs with loss. You need to configure model.compile(loss={<layer_name>: loss_type})
    model = tf.keras.Model(inputs=inputs,
                           outputs = [seg_output_prob, vin_output_prob, seg_category, vin_string],
                           )


    return model