print('Step 1 imports')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


from monet_boys.utils import set_distributed_training_strategy, discriminator_loss, generator_loss, calc_cycle_loss, identity_loss
from monet_boys.cyclegan import CycleGan
from monet_boys.generator import Generator
from monet_boys.discriminator import Discriminator
from monet_boys.data import load_dataset, read_tfrecord, decode_image

# Load data

AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 32
IMAGE_SIZE = [256, 256]

GCS_PATH = 'raw_data'

MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH +'/monet_tfrec/*.tfrec'))
PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH +'/photo_tfrec/*.tfrec'))

def decode_image(image):
    '''All the images for the competition are already sized to 256x256. As these images are RGB images, set the channel to 3.
    Additionally, we need to scale the images to a [-1, 1] scale'''
    image = tf.image.decode_jpeg(image, channels=3)
    #print(image.shape)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    #print(image.shape)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    #print(image.shape)
    print('decode_image')

    return image


def read_tfrecord(example):
    '''Because we are building a generative model,
    we don't need the labels or the image id so we'll only return the image
    from the TFRecord.'''

    tfrecord_format = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    print('read_tfrecord')
    return image


def load_dataset(filenames):
    '''Define a function to extract the image from the files.'''
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord)
    print('load_dataset')
    return dataset


monet_ds = load_dataset(MONET_FILENAMES).batch(1)
photo_ds = load_dataset(PHOTO_FILENAMES).batch(1)

final_ds = tf.data.Dataset.zip((monet_ds, photo_ds))

print(type(final_ds))

print(monet_ds.__dict__)


# Set up strategy to TPU if possible
print('Step 2 strategy')

strategy = set_distributed_training_strategy()

with strategy.scope():
    print('Step 3 generators and discrim')

    # Define generator and discriminators
    monet_generator = Generator() # transforms photos to Monet-esque paintings
    photo_generator = Generator() # transforms Monet paintings to be more like photos

    monet_discriminator = Discriminator() # differentiates real Monet paintings and generated Monet paintings
    photo_discriminator = Discriminator() # differentiates real photos and generated photos

    # Select optimizers
    monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    #Instantiate model
    cycle_gan_model = CycleGan(
        monet_generator, photo_generator, monet_discriminator, photo_discriminator
    )

    cycle_gan_model.compile(
        m_gen_optimizer = monet_generator_optimizer,
        p_gen_optimizer = photo_generator_optimizer,
        m_disc_optimizer = monet_discriminator_optimizer,
        p_disc_optimizer = photo_discriminator_optimizer,
        gen_loss_fn = generator_loss,
        disc_loss_fn = discriminator_loss,
        cycle_loss_fn = calc_cycle_loss,
        identity_loss_fn = identity_loss
    )

    cycle_gan_model.fit(
    tf.data.Dataset.zip((monet_ds, photo_ds)),
    epochs=1
    )
