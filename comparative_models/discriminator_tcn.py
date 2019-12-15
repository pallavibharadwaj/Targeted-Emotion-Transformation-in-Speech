from tensorflow.keras import backend
from tcn import TCN
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras.callbacks import callbacks
from tensorflow.keras import optimizers
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
            print(spect_file)

            features = np.load(spect_file)
            train_data.append(features)

            labels.append(int(spect_file.split('.')[0].split('-')[2])-1)
    return (np.array(train_data), np.array(labels))

def build_discriminator(time_steps, feature_size):
    print(backend.image_data_format())

    inputs = Input(batch_shape=(16, time_steps, feature_size))

    output1 = TCN(nb_filters=16,  dilations=[1, 2, 4, 8, 16, 32, 64], kernel_size=3, dropout_rate=0.1, return_sequences=True)(inputs)
    output2 = TCN(nb_filters=32, dilations=[1, 2, 4, 8, 16, 32, 64], kernel_size=3, dropout_rate=0.1, return_sequences=True)(output1)
    output3 = TCN(nb_filters=32, dilations=[1, 2, 4, 8, 16, 32, 64], kernel_size=3, dropout_rate=0.1, return_sequences=False)(output2)

    output = Dense(8, activation='softmax')(output3)

    model = Model(inputs=[inputs], outputs=[output])
    model.summary()

    return model

def train_discriminator(X_train, Y_train, model):
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['accuracy']
    )
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = callbacks.ModelCheckpoint('bestmodel_tcn.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')
    csv_logger = callbacks.CSVLogger('stats/Discriminator_stats_tcn.csv')

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
    print(X_train.shape, labels.shape)

    model = build_discriminator(X_train.shape[1], X_train.shape[2])
    train_discriminator(X_train, Y_train, model)

if __name__=='__main__':
    main()
