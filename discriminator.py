from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, Input, Embedding, concatenate
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import os

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("GPU version of TF not installed")

def load_data(datasets):
    train_data = []
    labels = []
    for dataset in datasets:
        for in_file in os.listdir(dataset):
            spect_file = os.path.join(dataset, in_file)

            features = np.load(spect_file)
            train_data.append(features)

            labels.append(int(spect_file.split('.')[0].split('-')[2])-1)
    return (np.array(train_data), np.array(labels))

def build_discriminator(time_steps, feature_size, time_block_count, features_block_count):
    spectrogram = Input(shape=(time_steps, feature_size))
    layer0 = Reshape((time_steps, feature_size, 1))(spectrogram)

    ## Embedding should be of size current_height * current_width
    required_size = time_steps * feature_size

    top = Input(shape=(1,))
    ## embedding for block position top
    pos_embedding = Embedding(time_block_count, required_size)(top)
    ## Reshape to additional channel
    top_channel = Reshape((time_steps, feature_size, 1))(pos_embedding)
    ## Concatenate to the input for the next layer
    layer0 = concatenate([layer0, top_channel], axis=3)

    left = Input(shape=(1,))
    ## embedding for block position left
    pos_embedding = Embedding(features_block_count, required_size)(left)
    ## Reshape to additional channel
    left_channel = Reshape((time_steps, feature_size, 1))(pos_embedding)
    ## Concatenate to the input for the next layer
    layer0 = concatenate([layer0, left_channel], axis=3)

    layer1 = Conv2D(32, kernel_size=(3,3), padding='same')(layer0)
    layer2 = MaxPooling2D(pool_size=(2,2))(layer1)

    layer3 = Conv2D(16, kernel_size=(3,3), padding='same')(layer2)
    layer4 = MaxPooling2D(pool_size=(2,2))(layer3)

    layer5 = Conv2D(16, kernel_size=(3,3), padding='same')(layer4)
    layer6 = MaxPooling2D(pool_size=(2,2))(layer5)

    layer7 = Flatten()(layer6)
    layer8 = Dense(16)(layer7)
    prediction = Dense(9, activation = 'softmax')(layer8)

    model = Model([spectrogram, top, left], prediction)
    model.summary()

    model.compile(
        loss='kld',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
