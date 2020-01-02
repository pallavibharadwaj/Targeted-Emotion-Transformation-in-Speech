from keras.models import Model
from keras import Input
from keras.backend import squeeze
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization, Activation, LeakyReLU, Lambda, Embedding, Dense, add, concatenate, Reshape, ZeroPadding2D, Cropping2D

## creates an down-sampling residual block of U-Net
## x: input to the residual block
## nb_channels: number of channels in the output of the block
def downsampling_res_block(x, nb_channels):
    res_path = BatchNormalization()(x)
    res_path = Conv2D(filters=nb_channels, kernel_size=(5, 3), padding='same')(res_path)
    res_path = MaxPooling2D(pool_size=(2, 2))(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Conv2D(filters=nb_channels, kernel_size=(5, 3), padding='same')(res_path)

    shortcut = Conv2D(nb_channels, kernel_size=(1, 1))(x)
    shortcut = MaxPooling2D(pool_size=(2, 2))(shortcut)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path

## creates an up-sampling residual block of U-Net
## x: input to the residual block
## nb_channels: number of channels in the output of the block
## branch: brancg from the corresponding down-sampling residual block
def upsampling_res_block(x, nb_channels, branch):
	x = UpSampling2D(size=(2, 2))(x)
	upsampled = concatenate([x, branch], axis=3)
	res_path = BatchNormalization()(upsampled)
	res_path = Conv2D(filters=nb_channels, kernel_size=(5, 3), padding='same')(res_path)
	res_path = BatchNormalization()(res_path)
	res_path = Conv2D(filters=nb_channels, kernel_size=(5, 3), padding='same')(res_path)

	shortcut = Conv2D(nb_channels, kernel_size=(1, 1))(upsampled)
	shortcut = BatchNormalization()(shortcut)

	res_path = add([shortcut, res_path])
	return res_path

## n_classes: number of classes of target emotions
## time_steps: length of each sequence (number of frames in each spectrogram)
## features_size: size of the feature vector at each point in the sequence (number of frequency bins in each spectrogram)
def build_generator(time_steps, features_size, time_block_count, features_block_count, n_classes=8):
	## input spectrogram
	spectrogram = Input(shape=(time_steps, features_size))
	## Reshape to 4D tensor
	x = Reshape((time_steps, features_size, 1))(spectrogram)

	## Embedding should be of size current_height * current_width
	required_size = time_steps * features_size

	top = Input(shape=(1,))
	## embedding for block position top
	pos_embedding = Embedding(time_block_count, required_size)(top)
	## Reshape to additional channel
	top_channel = Reshape((time_steps, features_size, 1))(pos_embedding)
	## Concatenate to the input for the next layer
	x = concatenate([x, top_channel], axis=3)

	left = Input(shape=(1,))
	## embedding for block position left
	pos_embedding = Embedding(features_block_count, required_size)(left)
	## Reshape to additional channel
	left_channel = Reshape((time_steps, features_size, 1))(pos_embedding)
	## Concatenate to the input for the next layer
	x = concatenate([x, left_channel], axis=3)

	## encoder
	u_path = Conv2D(filters=64, kernel_size=(5, 3), padding='same')(x)
	u_path = BatchNormalization()(u_path)

	u_path = Conv2D(filters=64, kernel_size=(5, 3), padding='same')(u_path)

	shortcut = Conv2D(filters=64, kernel_size=(1, 1))(x)
	shortcut = BatchNormalization()(shortcut)

	u_path = add([shortcut, u_path])
	branch1 = u_path

	u_path = downsampling_res_block(u_path, 128)
	branch2 = u_path

	u_path = downsampling_res_block(u_path, 256)
	branch3 = u_path

	## bridge
	u_path = downsampling_res_block(u_path, 512)

	## gets [batch_size, current_height, current_width, current_channel]
	current_shape = u_path.get_shape().as_list()
	required_size = current_shape[1] * current_shape[2]
	## creating an additional channel for target emotion
	## Embedding should be of size current_height * current_width
	## target_label: categorical label of target emotion
	target_label = Input(shape=(1,))
	## embedding for emotional class
	label_embedding = Embedding(n_classes, required_size)(target_label)
	## Reshape to additional channel
	target_channel = Reshape((current_shape[1], current_shape[2], 1))(label_embedding)
	## Concatenate to the input for the next layer
	u_path = concatenate([u_path, target_channel], axis=3)

	## decoder
	u_path = upsampling_res_block(u_path, 256, branch3)

	u_path = upsampling_res_block(u_path, 128, branch2)

	u_path = upsampling_res_block(u_path, 64, branch1)

	u_path = Conv2D(filters=1, kernel_size=(1, 1), activation='linear')(u_path)

	output = Lambda(lambda x: squeeze(x, axis=3))(u_path)

	generator = Model([spectrogram, top, left, target_label], output)
	
	return generator