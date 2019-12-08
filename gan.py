from generator import build_generator
from discriminator import build_discriminator, prep_data
from keras.layers import Reshape, Lambda
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.backend import expand_dims, clear_session
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf

import os

# def to_categorical(labels, num_classes):
#     categorical = np.zeros((labels.shape[0], num_classes), dtype=np.int32)
#     for i in range(labels.shape[0]):
#         categorical[i, labels[i]] = 1
#     return categorical

def build_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    for layer in discriminator.layers:
        layer.trainable = False
    # get noise and label inputs from generator model
    gen_input, top, left, gen_label = generator.input
    # get image output from the generator model
    gen_output = generator.output
    # Lambda to turn 4D
    # discriminator_input = Reshape((gen_output.shape[1], gen_output.shape[2], 1))(gen_output) #Lambda(lambda x:expand_dims(x, axis=-1))(gen_output)
    #print(discriminator_input.shape)
    # connect image output and label input from generator as inputs to discriminator
    gan_output = discriminator([gen_output, top, left])
    # define gan model as taking noise and label and outputting a classification
    gan = Model([gen_input, top, left, gen_label], gan_output)
# 
    # opt = Adam(lr=0.002, beta_1=0.5)
    gan.compile(loss='kld', optimizer='adam', metrics=['accuracy'])

    return gan

def generate_triplets(data, labels):
    indices = []
    targets = []
    classes = []

    # possible_targets = range(0, 8)
    
    for i in range(data.shape[0]):
        label = labels[i]
        possible_targets = np.delete(np.arange(0, 8), label)
        target = np.random.choice(possible_targets)

        indices.append(i)
        classes.append(label)
        targets.append(target)

        # for target in possible_targets:
        #     if label != target:
        #         indices.append(i)
        #         classes.append(label)
        #         targets.append(target)

    return np.stack([np.array(indices), np.array(classes), np.array(targets)], axis=-1)

def generate_fake_samples(x, labels, Y, X, generator):
    x_fake = []
    labels_fake = []
    for i in range(x.shape[0]):
        in_x = x[i, :, :]
        label = labels[i]

        possible_targets = np.delete(np.arange(0, 8), label)
        target = np.random.choice(possible_targets, 1)

        in_x = np.reshape(in_x, (1, in_x.shape[0], in_x.shape[1]))

        x_fake.append(generator.predict([in_x, Y[i], X[i], target]))
        labels_fake.append(8) ## label(fake) = 8

    return np.concatenate(x_fake, axis=0), np.array(labels_fake)

# train the generator and discriminator
def train(num_epochs=200, batch_size=50, validation_split=0.1, lr_discriminator=0.01, lr=0.002):
    data, labels = prep_data(['spectrograms/ravdess', 'spectrograms/savee'])
    print(data.shape, labels.shape)
    # triplets = generate_triplets(data, labels)

    time_steps = data.shape[1]
    features_size = data.shape[2]
    time_block_count = time_steps // 64
    feature_block_count = features_size // 64

    generator = build_generator(64, 64, time_block_count, feature_block_count)
    # generator.load_weights("generator/generator_epoch_8.h5")
    generator.summary()

    discriminator = build_discriminator(64, 64, time_block_count, feature_block_count)
    # opt = SGD(lr=lr_discriminator)
    # opt = Adam(lr=lr, beta_1=0.5)
    # discriminator.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # discriminator.load_weights("discriminator/discriminator_epoch_8.h5")
    discriminator.summary()

    # gan = build_gan(generator, discriminator)
    gan = build_gan(generator, discriminator)
   
    gan.summary()

    # manually enumerate epochs

    # min_accuracy = 0.0
    # min_d_loss_1 = 0.0
    # min_d_loss_2 = 0.0
    # sample_count = 0


    train_stats = open('Discriminator_stats.csv', 'a')
    train_stats.write('Epoch,D_Loss1,D_Loss2\n')
    train_stats.close()

    gen_stats = open('Generator_stats.csv', 'a')
    gen_stats.write('Epoch,G_Loss,G_Acc\n')
    gen_stats.close()

    batch_stats = open('Batch_stats.csv', 'a')
    batch_stats.write('Epoch,G_Loss,G_Acc\n')
    batch_stats.close()

    for epoch in range(2, num_epochs):
        d_loss_1 = []
        d_loss_2 = []

        # discriminator.trainable = True
        # discriminator.compile(
        #     loss='kld',
        #     optimizer='adam',
        #     metrics=['accuracy']
        # )

        batch_count = data.shape[0] // batch_size

        # enumerate batches over the training set
        for batch_index in range(batch_count):
            avg_d_loss1 = 0.0
            avg_d_loss2 = 0.0

            start = batch_index * batch_size

            if batch_index < batch_count - 1:
                batch_x = data[start:(start + batch_size), :, :]
                batch_labels = labels[start:(start + batch_size)]
                half_batch = int(batch_size / 2)
            else:
                batch_x = data[start:, :, :]
                batch_labels = labels[start:]
                half_batch = (data.shape[0] - start) // 2

            print(half_batch)

            x_real = batch_x[:half_batch, :, :]
            labels_real = batch_labels[:half_batch]

            for y in range(time_block_count):
                Y = np.repeat(y, x_real.shape[0]).reshape(-1, 1)
                for x in range(feature_block_count):
                    start_y = y * 64
                    start_x = x * 64
                    X = np.repeat(x, x_real.shape[0]).reshape(-1, 1)
                    d_loss1, _ = discriminator.train_on_batch([x_real[:, start_y:(start_y + 64), start_x:(start_x + 64)], Y, X], to_categorical(labels_real, num_classes=9))
                    # generate 'fake' examples
                    x_fake, labels_fake = generate_fake_samples(batch_x[half_batch:, start_y:(start_y + 64), start_x:(start_x + 64)], batch_labels[half_batch:], Y, X, generator)
                    # update discriminator model weights
                    d_loss2, _ = discriminator.train_on_batch([x_fake, Y, X], to_categorical(labels_fake, num_classes=9))
                    # summarize loss on this batch
                    avg_d_loss1 += (d_loss1 * x_real.shape[0])
                    avg_d_loss2 += (d_loss2 * x_fake.shape[0])
                   
                    print(y, x)
            print('Epoch %d, Batch %d/%d, d1=%.3f, d2=%.3f' %
                (epoch+1, batch_index+1, batch_count, avg_d_loss1 / (half_batch * time_block_count * feature_block_count), \
                avg_d_loss2 / (half_batch * time_block_count * feature_block_count))) #avg_g_loss/ sample_count, accuracy

            d_loss_1.append(avg_d_loss1 / (x_real.shape[0] * time_block_count * feature_block_count)) 
            d_loss_2.append(avg_d_loss2 / (x_fake.shape[0] * time_block_count * feature_block_count))

        # save the generator model
        dloss1 = sum(d_loss_1) / len(d_loss_1)
        dloss2 = sum(d_loss_2) / len(d_loss_2)

        train_stats = open('Discriminator_stats.csv', 'a')
        train_stats.write('%d,%.6f,%.6f\n'%(epoch + 1, dloss1, dloss2))
        train_stats.close()

        # if epoch == 0:
        #     min_d_loss_1 = dloss1
        #     discriminator.save('discriminator_loss1.h5')
        #     discriminator.save('discriminator_loss2.h5')
        # else:
        #     if min_d_loss_1 >= dloss1:
        #         min_d_loss_1 = dloss1
        #         discriminator.save('discriminator_loss1.h5')

        #     if min_d_loss_2 >= dloss2:
        #         min_d_loss_2 = dloss2
        #         discriminator.save('discriminator_loss2.h5')
        discriminator.save_weights('discriminator/discriminator_epoch_%d.h5'%(epoch + 1))

        triplets = generate_triplets(data, labels)
        np.random.shuffle(triplets)

        batch_count = int(triplets.shape[0] // batch_size)

        g_loss = []
        g_acc = []

        for batch_index in range(batch_count):
            avg_g_loss = 0.0
            avg_acc = 0.0

            start = batch_index * batch_size
            if batch_index < batch_count - 1:
                batch_x = data[triplets[start:(start + batch_size), 0], :, :]
                batch_labels = labels[triplets[start:(start + batch_size), 1]]
                batch_targets = triplets[start:(start + batch_size), 2]
            else:
                batch_x = data[triplets[start:,0], :, :]
                batch_labels = labels[triplets[start:, 1]]
                batch_targets = triplets[start:, 2]

            for y in range(time_block_count):
                Y = np.repeat(y, batch_x.shape[0]).reshape(-1, 1)
                for x in range(feature_block_count):
                    start_y = y * 64
                    start_x = x * 64
                    X = np.repeat(x, batch_x.shape[0]).reshape(-1, 1)

                    # update the generator via the discriminator's error
                    loss, acc = gan.train_on_batch([batch_x[:, start_y:(start_y + 64), start_x:(start_x + 64)], Y, X, batch_targets], to_categorical(batch_targets, num_classes=9))

                    avg_g_loss += (loss * batch_x.shape[0])
                    print(loss, batch_x.shape[0], time_block_count, feature_block_count, acc)
                    avg_acc += (acc * batch_x.shape[0])

                    print(y, x)

            print('Epoch %d, Batch %d/%d, g_loss=%.3f, acc=%.3f' %
                (epoch+1, batch_index+1, batch_count, avg_g_loss // (batch_x.shape[0] * time_block_count * feature_block_count), \
                avg_acc / (batch_x.shape[0] * time_block_count * feature_block_count)))
            
            batch_stats = open('Batch_stats.csv', 'a')
            batch_stats.write('%d,%d,%f,%f\n'%(epoch+1, batch_index+1, avg_g_loss // (batch_x.shape[0] * time_block_count * feature_block_count), \
                avg_acc / (batch_x.shape[0] * time_block_count * feature_block_count)))
            batch_stats.close()

            g_loss.append(avg_g_loss / (batch_x.shape[0] * time_block_count * feature_block_count)) 
            g_acc.append(avg_acc / (batch_x.shape[0] * time_block_count * feature_block_count))

        # save the generator model
        gen_loss = sum(g_loss) / len(g_loss)
        gen_acc = sum(g_acc) / len(g_acc)

        gen_stats = open('Generator_stats.csv', 'a')
        gen_stats.write('%d,%.6f,%.6f\n'%(epoch + 1, gen_loss, gen_acc))
        gen_stats.close()

        generator.save_weights('generator/generator_epoch_%d.h5'%(epoch + 1))


    train_stats.close()
    gen_stats.close()
    batch_stats.close()

def main():
    train(num_epochs=15)

if __name__=='__main__':
    main()
