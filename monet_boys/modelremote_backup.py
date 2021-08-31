import os
from google.cloud import storage
from google.cloud.storage import bucket
from dotenv import load_dotenv
import os
import json
import requests
import streamlit as st
import tensorflow as tf
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import PIL
import SessionState


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../raw_data/batch-672-gan-monet.json'

PROJECT = "bucket-monet-gan"  # change for your GCP project
REGION = "europe-west1"
MODEL = "cezanne_v1_1"


st.title("Welcome to Monet Vision 🖌📸")
st.header("Turn your picture into art!")


########################################################################
#  Upsample and downsample                                             #
########################################################################
OUTPUT_CHANNELS = 3


def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(
            gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result

#####################################
# Generator function                #
#####################################


def Generator():
    inputs = layers.Input(shape=[256, 256, 3])

    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')  # (bs, 256, 256, 3)

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

    return keras.Model(inputs=inputs, outputs=x)


########################################################################
#  Import model                                                        #
########################################################################
cezanne_generator = Generator()
#cezanne_generator.load_weights(MODEL)


######################################################
# predict JSON from google models                    #
######################################################


def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to Tensors.
        version (str): version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    model_path = "projects/{}/models/{}".format(project, model)
    if version is not None:
        model_path += "/versions/{}".format(version)

    # Create ML engine resource endpoint and input data
    ml_resource = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
    # turn input into list (ML Engine wants JSON)
    instances_list = instances.numpy().tolist()

    input_data_json = {"signature_name": "serving_default",
                       "instances": instances_list}

    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    return response


####################################
# process image                    #
###################################
# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(uploaded_file, img_shape=256):

    IMAGE_SIZE = [256, 256]
    image = PIL.Image.open(session_state.uploaded_image)
    resized_image = image.resize((256, 256))
    input_tensor = tf.io.read_file(resized_image)

    imagetensor = tf.io.decode_image(uploaded_file, channels=3)
    imagetensor = tf.image.resize(img, [img_shape, img_shape])
    imagetensor = (tf.cast(imagetensor, tf.float32) / 127.5) - 1
    imagetensor = tf.reshape(imagetensor, [*IMAGE_SIZE, 3])
    imagetensor = tf.expand_dims(imagetensor, axis=0)
    return imagetensor


################################################################
# Make prediction
################################################################
def make_prediction():

    imagetensor = load_and_prep_image(uploaded_file)
    cezanne_prediction = cezanne_generator(
        imagetensor, training=False)[0].numpy()
    cezanne_prediction = (cezanne_prediction * 127.5 + 127.5).astype(np.uint8)
    st.image(cezanne_prediction)


########################################################################
#  Upload file with button                                             #
########################################################################
uploaded_file = st.file_uploader(label="Upload an image of your choice",
                                 type=["png", "jpeg", "jpg"])

# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11
session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True

# And if they did...
if session_state.pred_button:
    st.write(type(uploaded_file))
    session_state.image = make_prediction()

    #session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(
    #    session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    #st.write(f"Prediction: {session_state.pred_class}, \
    #           Confidence: {session_state.pred_conf:.3f}")
