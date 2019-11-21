from keras.models import Model
from keras.layers import Dense, Dropout, LSTM
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
    print("Please install GPU version of TF")

MAXLEN = 0

def prep_data(in_folder):
    train_data = []
    labels = []

    for in_file in os.listdir(in_folder):
        spect_file = os.path.join(in_folder, in_file)
        
        features = np.loadtxt(spect_file)
        labels.append(int(spect_file.split('-')[2]))

        # finding with max length
        global MAXLEN
        if(MAXLEN < features.shape[1]):
            MAXLEN = features.shape[1]

        train_data.append(features)
    # pad to longest audio
    train_data = np.array([np.pad(audio, ((0,0), (0, MAXLEN-audio.shape[1])), 'constant').transpose() for audio in train_data])

    return (train_data, np.array(labels))

def train_discriminator(X_train, Y_train):
    model_lstm = Sequential()

    model_lstm.add(LSTM(514, input_shape = (MAXLEN, 514), activation = 'relu'))
    model_lstm.add(Dropout(0.3))

    model_lstm.add(Dense(256, activation = 'relu'))
    model_lstm.add(Dropout(0.1))

    model_lstm.add(Dense(128, activation = 'relu'))
    model_lstm.add(Dropout(0.1))

    model_lstm.add(Dense(64, activation = 'relu'))
    model_lstm.add(Dropout(0.1))

    model_lstm.add(Dense(32, activation = 'relu'))
    model_lstm.add(Dropout(0.1))

    model_lstm.add(Dense(16, activation = 'relu'))
    model_lstm.add(Dropout(0.1))

    model_lstm.add(Dense(9, activation = 'softmax'))
    sgd = optimizers.SGD(lr=0.01)

    model_lstm.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = callbacks.ModelCheckpoint('bestmodel.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')

    history = model_lstm.fit(
        X_train,
        Y_train,
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
        validation_split = 0.1,
        epochs = 200,
        batch_size = 32
    )

def main():
    (X_train, labels) = prep_data('spectrograms/ravdess')

    Y_train = to_categorical(labels)   # labels to one-hot
    print(X_train.shape, Y_train.shape)

    train_discriminator(X_train, Y_train)

if __name__=='__main__':
    main()

