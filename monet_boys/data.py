import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import numpy as np
import re

'''We want to keep our photo dataset and our Monet dataset separate.
    First, load in the filenames of the TFRecords.'''

BATCH_SIZE = 32
IMAGE_SIZE = [256, 256]

GCS_PATH = '../raw_data'

MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH +'/monet_tfrec/*.tfrec'))
PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH +'/photo_tfrec/*.tfrec'))

def decode_image(image):
    '''All the images for the competition are already sized to 256x256. As these images are RGB images, set the channel to 3.
    Additionally, we need to scale the images to a [-1, 1] scale'''
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
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
    return image


def load_dataset(filenames, labeled=True, ordered=False):
    '''Define a function to extract the image from the files.'''
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset

def draw_examples(dataset):
    '''test to draw examples in a notebook to see if the loading was successful'''
    monet_ds = load_dataset(MONET_FILENAMES, labeled=True).batch(1)
    photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True).batch(1)
    example_monet = next(iter(monet_ds))
    example_photo = next(iter(photo_ds))
    plt.subplot(121)
    plt.title('Photo')
    plt.imshow(example_photo[0] * 0.5 + 0.5)
    plt.subplot(122)
    plt.title('Monet')
    plt.imshow(example_monet[0] * 0.5 + 0.5)




if __name__ == '__main__':
    print(len(MONET_FILENAMES))
    print(len(PHOTO_FILENAMES))
