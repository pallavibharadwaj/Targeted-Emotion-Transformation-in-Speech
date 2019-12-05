from keras import backend
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, LeakyReLU, Input, Embedding, concatenate
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import callbacks
from keras import optimizers
import tensorflow as tf
import numpy as np
import os

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("GPU version of TF not installed")

def prep_data(datasets):
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
    # spectrogram = tf.placeholder(tf.float32, shape=(None, time_steps, feature_size))
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
    #model.add(LeakyReLU(alpha=0.01))
    layer2 = MaxPooling2D(pool_size=(2,2))(layer1)

    layer3 = Conv2D(16, kernel_size=(3,3), padding='same')(layer2)
    #model.add(LeakyReLU(alpha=0.05))
    layer4 = MaxPooling2D(pool_size=(2,2))(layer3)

    layer5 = Conv2D(16, kernel_size=(3,3), padding='same')(layer4)
    #model.add(LeakyReLU(alpha=0.05))
    layer6 = MaxPooling2D(pool_size=(2,2))(layer5)

    layer7 = Flatten()(layer6)
    layer8 = Dense(16)(layer7)
    prediction = Dense(9, activation = 'softmax')(layer8)
    # prediction = Dropout(0.1)(layer9)

    model = Model([spectrogram, top, left], prediction)
    model.summary()

    # sgd = optimizers.SGD(lr=0.01)

    model.compile(
        loss='kld',
        optimizer='adam',
        metrics=['accuracy']
    )
    #opt = optimizers.Adam(lr=0.002, beta_1=0.5)
    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def train_discriminator(X_train, Y_train, model):
    #X_train = X_train.reshape(1524, 1230, 514, 1)
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = callbacks.ModelCheckpoint('bestmodel.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')

    history = model.fit(
        X_train,
        Y_train,
        callbacks=[mcp_save, reduce_lr_loss],
        validation_split = 0.1,
        epochs = 300,
        batch_size = 16
    )

def main():
    (x_train, labels) = prep_data(['spectrograms/ravdess', 'spectrograms/savee'])

    y_train = to_categorical(labels, num_classes=9)   # labels to one-hot
    print(x_train.shape, y_train.shape)
    np.savetxt('labels', y_train)

    model = build_discriminator(1230, 514)
    model.train_on_batch(x_train, y_train)
    #train_discriminator(x_train, y_train, model)

if __name__=='__main__':
    main()

