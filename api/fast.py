import uvicorn
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytz
import joblib


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import PIL

## Global variables

OUTPUT_CHANNELS = 3

## Useful functions definitions

def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

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

def Generator():
    inputs = layers.Input(shape=[256,256,3])

    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') # (bs, 256, 256, 3)

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



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello woorld"}

@app.get("/predict")
def predict(image_name, painter_name):

    # Import Cycle GAN trained model (First in local)
    painter_path = '/home/amineea/code/florentiino/monet_boys/raw_data/weights/output_cezanne/cezanne_model.h5'

    # Load image, resize if necessary
    IMAGE_SIZE = [256, 256]

    image_path = f'raw_data/images/{image_name}.jpg'

    image = PIL.Image.open(image_path)

    old_size = image.size

    if image.size != (256,256):
        image = image.resize((256,256))

    new_image_path = image_path.replace('.', '_new.')

    image.save(new_image_path)

    # Load (resized) image as tensor and apply preprocessing (rescaling)

    input_tensor = tf.io.read_file(new_image_path)

    imagetensor = tf.image.decode_jpeg(input_tensor, channels=3)
    imagetensor = (tf.cast(imagetensor, tf.float32) / 127.5) - 1

    imagetensor = tf.reshape(imagetensor, [*IMAGE_SIZE, 3])
    imagetensor = tf.expand_dims(imagetensor, axis=0)

    # Import and load trained model

    painter_path = f'raw_data/weights/output_{painter_name}/{painter_name}_model.h5'

    painter_generator = Generator()

    start_load = time.time()

    painter_generator.load_weights(painter_path)

    end_load = time.time()

    loading_time = end_load - start_load

    # Image transformation

    painter_prediction = painter_generator(imagetensor, training=False)[0].numpy()
    painter_prediction = (painter_prediction * 127.5 + 127.5).astype(np.uint8)

    transformed_image = PIL.Image.fromarray(painter_prediction)
    transformed_image.save('raw_data/images/' + image_name + '_' + painter_name + ".jpg")

    end_load_2 = time.time()

    loading_transforming_time = end_load_2 - start_load


    #Retrun dictionary

    return_dict = dict(
        imsize=old_size,
        new_name=new_image_path,
        newsize=image.size,
        typetensor=str(imagetensor.shape),
        painterpath=painter_path,
        time_loading = loading_time,
        tome_loading_transorming = loading_transforming_time
    )

    return return_dict


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)



    # # create datetime object from user provided date
    # pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # # localize the user provided datetime with the NYC timezone
    # eastern = pytz.timezone("US/Eastern")
    # localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # # convert the user datetime to UTC
    # utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

    # # format the datetime as expected by the pipeline
    # formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    # # fixing a value for the key, unused by the model
    # # in the future the key might be removed from the pipeline input
    # # eventhough it is used as a parameter for the Kaggle submission
    # key = "2013-07-06 17:18:00.000000119"

    # # build X ⚠️ beware to the order of the parameters ⚠️
    # X = pd.DataFrame(dict(
    #     key=[key],
    #     pickup_datetime=[formatted_pickup_datetime],
    #     pickup_longitude=[float(pickup_longitude)],
    #     pickup_latitude=[float(pickup_latitude)],
    #     dropoff_longitude=[float(dropoff_longitude)],
    #     dropoff_latitude=[float(dropoff_latitude)],
    #     passenger_count=[int(passenger_count)]))

    # # pipeline = get_model_from_gcp()
    # pipeline = joblib.load('model.joblib')

    # # make prediction
    # results = pipeline.predict(X)

    # # convert response from numpy to python type
    # pred = float(results[0])

    #return dict(prediction=pred)

