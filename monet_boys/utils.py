import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


### Setting up of strategy

def set_distributed_training_strategy():
    '''
    Sets the tensorflow distributed training strategy to TPU if possible
    '''
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Device:', tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.get_strategy()
    print('Number of replicas:', strategy.num_replicas_in_sync)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    print(tf.__version__)
    return strategy


### Generator and discriminator layers

'''
1. Downsampling
The downsample reduces the 2D dimensions, the width and height, of the image
by the stride within a 2D convolution layer
The stride is the length of the step the filter takes. Choosing a stride of 2 means that
the filter is applied to every other pixel, reducing the weight and height by 2.

The downsample layer will (optionally) use instance normalization (i.e. each sample output from layer n will be
normalised as it is passed as an input to layer n+1) --> requires TensorFlow addons
'''
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

'''
2. Upsampling
The upsampler will do the opposite of the downsampler and increase the dimensions of the image
by using a transposed 2D convolution layer of stride 2 instead of a 2D convilution layer. The upsampler
will use instance normalization by default, but will optionally apply dropout (to prevent overfitting)
'''
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

### Loss functions definition

def discriminator_loss(real, generated):#, strategy):
    '''
    Discriminator loss function compares real images to a matrix of 1s and fake images to a matrix of 0s.
    The perfect discriminator will output all 1s for real images and all 0s for fake images.
    Discriminator loss outputs the average of the real and generated loss
    '''
    # with strategy.scope():
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5

def generator_loss(generated):#, strategy):
    '''Generator wants to fool the discriminator into thinking the generated image is real.
    The perfect generator will have the discriminator output only 1s.
    Thus, it compares the generated image to a matrix of 1s to find the loss.
    '''
    #with strategy.scope():
    return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image, LAMBDA):#, strategy):
    '''
    Objective : get twice transformed photo as similar as possible to original real photo
    Cycle consistency loss is defined as finding the average of their difference
    Minimizing this loss will mean that cycled image will become extremely close to original image
    '''
    #with strategy.scope():
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1

def identity_loss(real_image, same_image, LAMBDA):#, strategy):
    '''
    Objective : get self_generated image as  similar as possible to original real image
    (ex : monet -> monet generator shoud generate a monet as close as possible to original monet)
    Identity lossis defined as finding the average of their difference
    Minimizing this loss will mean that generated image will become extremely close to original image
    '''
    #with strategy.scope():
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss



