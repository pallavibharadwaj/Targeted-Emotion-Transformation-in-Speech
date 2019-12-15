from keras.models import Model
from keras.layers import Dense, Dropout, LSTM
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import callbacks
from keras import optimizers
from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
import os

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

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

def build_discriminator(time_steps, feature_size):
    print(backend.image_data_format())

    model = Sequential()
    model.add(LSTM(32, activation = 'relu'))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.1))

    model.add(Dense(16))
    model.add(Dropout(0.1))

    model.add(Dense(8, activation = 'softmax'))
    
    return model

def train_discriminator(X_train, Y_train, model):
    sgd = optimizers.SGD(lr=0.005)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = callbacks.ModelCheckpoint('bestmodel.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')
    csv_logger = callbacks.CSVLogger('Discriminator_stats_lstm.csv')

    history = model.fit(
        X_train,
        Y_train,
        callbacks=[mcp_save, reduce_lr_loss, csv_logger],
        validation_split = 0.1,
        epochs = 100,
        batch_size = 16
    )

def main():
    (X_train, labels) = load_data(['spectrograms/ravdess', 'spectrograms/savee'])
    Y_train = to_categorical(labels)   # labels to one-hot
    print(X_train.shape, Y_train.shape)

    model = build_discriminator(X_train.shape[1], X_train.shape[2])
    train_discriminator(X_train, Y_train, model)

if __name__=='__main__':
    main()

