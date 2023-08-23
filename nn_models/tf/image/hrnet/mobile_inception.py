import tensorflow as tf
def base_block(x, stride=1, multiplier=5, output_filer=32):
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=multiplier,
        kernel_size=(3,3),
        padding="same",
        strides=stride,
        use_bias=False,
        activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = base_cba(x, output_filer, kernel_size=1, strides=1, activation="relu")
    return x

def base_cba(x, out_filters, kernel_size=1, strides=1, activation="relu"):
    x = tf.keras.layers.Conv2D(filters=out_filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding="same",
                               use_bias=False,
                               activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    if activation is not None:
        x = tf.keras.layers.Activation("relu")(x)
    return x



def get_mobile_inception(x, output_dim=36):
    x = base_cba(x, out_filters=16, kernel_size=3, strides=2)
    x = base_block(x, 2, multiplier=3, output_filer=32)
    x = base_block(x, 2, multiplier=3, output_filer=64)
    x = base_block(x, 2, multiplier=3, output_filer=128)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(output_dim)(x)
    return x



