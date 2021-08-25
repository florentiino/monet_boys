"""
generator.py -- Generator()

A generator is comprised of a downsample and an upsample method.

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

3. Generator building with long skip connection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import downsample, upsample

# Global variables

OUTPUT_CHANNELS = 3

def Generator():
    """ downsamples then upsamples an image
    while establishing skip connections (to handle vanishing gradient) from downsample output
    to upsample layer in symetrical fashion
    --> returns a Keras model
    """

    # Define input layer + shape
    inputs = layers.Input(shape=[256,256,3])

    # Instantiate down stack using downsample function previously defined
    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)

        downsample(512, 4), # (bs, 1, 1, 512) # Last layer of the down stack, won't have skip connection with up_stack
    ]

    # Instantiate down stack using downsample function previously defined
    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    # Add last layer that will uspample the previous layer to a final 256x256x3 image
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') # (bs, 256, 256, 3)


    ## Establish skip connections
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    # Return a keras model
    return keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":

    # Instantiate one Generator
    a = Generator()
    print(a.summary())
