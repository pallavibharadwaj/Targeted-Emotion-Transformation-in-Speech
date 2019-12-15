from keras import backend
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, Input
from keras.utils import to_categorical
from keras.callbacks import callbacks
from keras import optimizers
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

def build_discriminator(time_steps, feature_size):
    spectrogram = Input(shape=(time_steps, feature_size))

    layer0 = Reshape((time_steps, feature_size, 1))(spectrogram)
    layer1 = Conv2D(32, kernel_size=(3,3), padding='same')(layer0)
    layer2 = MaxPooling2D(pool_size=(4,4))(layer1)

    layer3 = Conv2D(16, kernel_size=(3,3), padding='same')(layer2)
    layer4 = MaxPooling2D(pool_size=(4,4))(layer3)

    layer5 = Conv2D(16, kernel_size=(3,3), padding='same')(layer4)
    layer6 = MaxPooling2D(pool_size=(4,4))(layer5)

    layer7 = Flatten()(layer6)
    layer8 = Dense(16)(layer7)
    prediction = Dense(9, activation = 'softmax')(layer8)

    model = Model(spectrogram, prediction)
    model.summary()

    return model


def train_discriminator(X_train, Y_train, model):
    sgd = optimizers.SGD(lr=0.01)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = callbacks.ModelCheckpoint('bestmodel.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')
    csv_logger = callbacks.CSVLogger('Discriminator_stats_cnn.csv')
    history = model.fit(
        X_train,
        Y_train,
        callbacks=[mcp_save, reduce_lr_loss, csv_logger],
        validation_split = 0.1,
        epochs = 100,
        batch_size = 16
    )

def main():
    (x_train, labels) = load_data(['spectrograms/ravdess', 'spectrograms/savee'])

    y_train = to_categorical(labels, num_classes=9)   # labels to one-hot

    model = build_discriminator(x_train.shape[1], x_train.shape[2])
    train_discriminator(x_train, y_train, model)

if __name__=='__main__':
    main()
