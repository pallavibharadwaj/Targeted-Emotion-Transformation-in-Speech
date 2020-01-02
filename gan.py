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

def build_gan(generator, discriminator):
    ## make weights in the discriminator not trainable
    for layer in discriminator.layers:
        layer.trainable = False
    ## get input spectrogram, block position(top, left) and target label from generator model
    ## top: least time step in the block // 64
    ## left: least frequency bin in the block // 64
    gen_input, top, left, gen_label = generator.input
    ## get output spectrogram from the generator model
    gen_output = generator.output

    ## connect spectrogram output and vlock position from generator as inputs to discriminator
    gan_output = discriminator([gen_output, top, left])
    ## define gan model as taking input spectrogram and target label and outputting a classification
    gan = Model([gen_input, top, left, gen_label], gan_output)

    gan.compile(loss='kld', optimizer='adam', metrics=['accuracy'])

    return gan

def generate_triplets(data, labels):
    indices = []
    targets = []
    classes = []
    
    for i in range(data.shape[0]):
        label = labels[i]

        ## generate a target label that is not the same as the ground truth label for the given data
        possible_targets = np.delete(np.arange(0, 8), label)
        target = np.random.choice(possible_targets)

        indices.append(i)
        classes.append(label)
        targets.append(target)

    return np.stack([np.array(indices), np.array(classes), np.array(targets)], axis=-1)

def generate_fake_samples(x, labels, Y, X, generator):
    x_fake = []
    labels_fake = []
    for i in range(x.shape[0]):
        in_x = x[i, :, :]
        label = labels[i]

        ## generate fake sample by passing in_x to the generator
        ## in_x is a half-batch of spectrogram with the same block position (top, left).
        ## target is chosen at random from a list of possible target labels, excepting the ground truth label of in_x.
        possible_targets = np.delete(np.arange(0, 8), label)
        target = np.random.choice(possible_targets, 1)
        in_x = np.reshape(in_x, (1, in_x.shape[0], in_x.shape[1]))
        x_fake.append(generator.predict([in_x, Y[i], X[i], target]))

        ## the generated sample is labelled 'fake'.
        labels_fake.append(8) ## label(fake) = 8

    return np.concatenate(x_fake, axis=0), np.array(labels_fake)

## train the generator and discriminator
def train(num_epochs=200, batch_size=50, block_size=64):
    data, labels = prep_data(['spectrograms/ravdess', 'spectrograms/savee'])

    time_steps = data.shape[1]
    features_size = data.shape[2]
    time_block_count = time_steps // block_size
    feature_block_count = features_size // block_size

    generator = build_generator(block_size, block_size, time_block_count, feature_block_count)
    generator.summary()

    discriminator = build_discriminator(block_size, block_size, time_block_count, feature_block_count)
    discriminator.summary()

    gan = build_gan(generator, discriminator)
   
    gan.summary()

    train_stats = open('Discriminator_stats.csv', 'a')
    train_stats.write('Epoch,D_Loss1,D_Loss2\n')
    train_stats.close()

    gen_stats = open('Generator_stats.csv', 'a')
    gen_stats.write('Epoch,G_Loss,G_Acc\n')
    gen_stats.close()

    batch_stats = open('Batch_stats.csv', 'a')
    batch_stats.write('Epoch,G_Loss,G_Acc\n')
    batch_stats.close()

    for epoch in range(num_epochs):
    	## the discriminator's weights are only updated every 5 epochs
        if epoch % 5 == 0:
            d_loss_1 = []
            d_loss_2 = []

            batch_count = data.shape[0] // batch_size

            ## enumerate batches over the training set
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

                x_real = batch_x[:half_batch, :, :]
                labels_real = batch_labels[:half_batch]

                for y in range(time_block_count):
                    Y = np.repeat(y, x_real.shape[0]).reshape(-1, 1)
                    for x in range(feature_block_count):
                        start_y = y * block_size
                        start_x = x * block_size
                        X = np.repeat(x, x_real.shape[0]).reshape(-1, 1)
                        ## train the discriminator on half a batch of gound truth data
                        d_loss1, _ = discriminator.train_on_batch([x_real[:, start_y:(start_y + block_size), start_x:(start_x + block_size)], Y, X], to_categorical(labels_real, num_classes=9))
                        ## generate 'fake' examples with half the batch
                        x_fake, labels_fake = generate_fake_samples(batch_x[half_batch:, start_y:(start_y + block_size), start_x:(start_x + block_size)], batch_labels[half_batch:], Y, X, generator)
                        ## update discriminator model weights using 'fake' samples
                        d_loss2, _ = discriminator.train_on_batch([x_fake, Y, X], to_categorical(labels_fake, num_classes=9))
                        ## summarize loss on this batch
                        avg_d_loss1 += (d_loss1 * x_real.shape[0])
                        avg_d_loss2 += (d_loss2 * x_fake.shape[0])
                       
                print('Epoch %d, Batch %d/%d, d1=%.3f, d2=%.3f' %
                    (epoch+1, batch_index+1, batch_count, avg_d_loss1 / (half_batch * time_block_count * feature_block_count), \
                    avg_d_loss2 / (half_batch * time_block_count * feature_block_count))) #avg_g_loss/ sample_count, accuracy

                d_loss_1.append(avg_d_loss1 / (x_real.shape[0] * time_block_count * feature_block_count)) 
                d_loss_2.append(avg_d_loss2 / (x_fake.shape[0] * time_block_count * feature_block_count))

            dloss1 = sum(d_loss_1) / len(d_loss_1)
            dloss2 = sum(d_loss_2) / len(d_loss_2)

            train_stats = open('Discriminator_stats.csv', 'a')
            train_stats.write('%d,%.6f,%.6f\n'%(epoch + 1, dloss1, dloss2))
            train_stats.close()

            ## save the discriminator model
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
                    start_y = y * block_size
                    start_x = x * block_size
                    X = np.repeat(x, batch_x.shape[0]).reshape(-1, 1)

                    # update the generator via the discriminator's error
                    loss, acc = gan.train_on_batch([batch_x[:, start_y:(start_y + block_size), start_x:(start_x + block_size)], Y, X, batch_targets], to_categorical(batch_targets, num_classes=9))

                    avg_g_loss += (loss * batch_x.shape[0])
                    avg_acc += (acc * batch_x.shape[0])

            print('Epoch %d, Batch %d/%d, g_loss=%.3f, acc=%.3f' %
                (epoch+1, batch_index+1, batch_count, avg_g_loss // (batch_x.shape[0] * time_block_count * feature_block_count), \
                avg_acc / (batch_x.shape[0] * time_block_count * feature_block_count)))
            
            batch_stats = open('Batch_stats.csv', 'a')
            batch_stats.write('%d,%d,%f,%f\n'%(epoch+1, batch_index+1, avg_g_loss // (batch_x.shape[0] * time_block_count * feature_block_count), \
                avg_acc / (batch_x.shape[0] * time_block_count * feature_block_count)))
            batch_stats.close()

            g_loss.append(avg_g_loss / (batch_x.shape[0] * time_block_count * feature_block_count)) 
            g_acc.append(avg_acc / (batch_x.shape[0] * time_block_count * feature_block_count))

        gen_loss = sum(g_loss) / len(g_loss)
        gen_acc = sum(g_acc) / len(g_acc)

        gen_stats = open('Generator_stats.csv', 'a')
        gen_stats.write('%d,%.6f,%.6f\n'%(epoch + 1, gen_loss, gen_acc))
        gen_stats.close()

        # save the generator model
        generator.save_weights('generator/generator_epoch_%d.h5'%(epoch + 1))


    train_stats.close()
    gen_stats.close()
    batch_stats.close()

def main():
    train(num_epochs=15)

if __name__=='__main__':
    main()
