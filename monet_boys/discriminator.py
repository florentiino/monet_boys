import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


"""
discirminator.py -- class Discriminator()
1. Downsampling
The downsample reduces the 2D dimensions, the width and height, of the image
by the stride within a 2D convolution layer
The stride is the length of the step the filter takes. Choosing a stride of 2 means that
the filter is applied to every other pixel, reducing the weight and height by 2.
The downsample layer will (optionally) use instance normalization (i.e. each sample output from layer n will be
normalised as it is passed as an input to layer n+1) --> requires TensorFlow addons
2. Upsampling
The upsampler will do the opposite of the downsampler and increase the dimensions of the image
by using a transposed 2D convolution layer of stride 2 instead of a 2D convilution layer. The upsampler
will use instance normalization by default, but will optionally apply dropout (to prevent overfitting)
"""

# Global variables SUGGESTION: lets move sampling to utilities.py for cleaner code

OUTPUT_CHANNELS = 3


def downsample(filters, size, apply_instancenorm=True):
    """ defines and returns a downsampler layer with parameters :
    filter (dimensionality of output space)
    size (kernel size)
    (+ optionally applies instance normalization)
    """
    #Initialize random distributions for the kernel initializer and the instance normalizer
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    # Instantiate model
    result = keras.Sequential()

    # Add 2D convolutional layer w. stride 2 and padding
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    # Apply instance normalization
    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(
            gamma_initializer=gamma_init))

    # Add a last Leaky ReLU layer
    result.add(layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    """ defines and returns an upsampler layer with parameters :
        filter (dimensionality of output space)
        size (kernel size)
        (+ optionally applies droupout)
    """
    #Initialize random distributions for the kernel initializer and the instance normalizer
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    # Instantiate model
    result = keras.Sequential()

    # Add transposed 2D convolutional layer w. stride 2 and padding
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    # Apply instance normalization
    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    # Apply droupout
    if apply_dropout:
        result.add(layers.Dropout(0.5))

    # Add a last ReLU layer
    result.add(layers.ReLU())

    return result


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)


if __name__ == "__main__":

    # Instantiate one Discriminator
    d = Discriminator()
    print(d.summary())
