import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import numpy as np
import re
import os


'''We want to keep our photo dataset and our Monet dataset separate.
    First, load in the filenames of the TFRecords.'''

AUTOTUNE = tf.data.experimental.AUTOTUNE



BATCH_SIZE = 32
IMAGE_SIZE = [256, 256]

GCS_PATH = 'raw_data'

MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH +'/monet_tfrec/*.tfrec'))
PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH +'/photo_tfrec/*.tfrec'))

CEZANNE_FILENAMES = tf.io.gfile.glob(str(GCS_PATH +'/trainA/*.jpg'))


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

## Read images from tfrec directory
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

## Read images from jpg directory
def read_image(file):
    '''Alternative method for reading images from a jpg.'''

    example = tf.io.read_file(file)
    image = decode_image(example)
    print('read_image')
    return image

def load_jpg_dataset(filenames):
    '''Define a function to extract the image from the files in a jpg format.'''
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(read_image)

    print('load_jpg_dataset')
    return dataset


if __name__ == '__main__':
    print(len(CEZANNE_FILENAMES))
    # print(len(PHOTO_FILENAMES))

    cezanne_ds = load_jpg_dataset(CEZANNE_FILENAMES)
    example_cezanne = next(iter(cezanne_ds)).numpy()
    img = example_cezanne
    example_cezanne = (example_cezanne * 127.5 + 127.5).astype(np.uint8)
    img = PIL.Image.fromarray(example_cezanne)
    img.save(GCS_PATH+"/test/" + str('test_cezanne') + ".jpg")

    # monet_ds = load_dataset(MONET_FILENAMES).batch(1)
    # example_monet = next(iter(monet_ds)).numpy()
    # print(example_monet.shape)
    # print(type(example_monet))
    # example_monet = (example_monet * 127.5 + 127.5).astype(np.uint8)
    # print(example_monet.shape)
    # img = example_monet
    # img=img[0]
    # img = PIL.Image.fromarray(img)
    # img.save(GCS_PATH+"/test/" + str('test_monet') + ".jpg")



