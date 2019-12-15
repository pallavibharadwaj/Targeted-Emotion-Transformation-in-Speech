import numpy as np
from librosa import load, stft, istft
from librosa.output import write_wav
from generator import build_generator
import argparse

time_steps = 1230
feature_size = 514

max_time_steps = time_steps + (64-(time_steps % 64))
max_feature_size = feature_size + (64-(feature_size % 64))

emotions = {
    'neutral': 1,
    'calm': 2,
    'happy': 3,
    'sad': 4,
    'angry': 5,
    'fear': 6,
    'disgust': 7,
    'surprise': 8,
}

def reconstruct(spectrogram):
    # remove the padding from the speech
    spectrogram = spectrogram[:feature_size, :feature_size].transpose()
    # including the real and imaginary components
    spectrogram = spectrogram[:257, :] + 1j * spectrogram[257:, :]
    # re-construct audio from spectrogram
    wav = istft(spectrogram)
    write_wav("target.wav", wav, sr=44100)

def generate_spectrogram(inputs):
    signal, sr = load(inputs, sr =44100)
    spectrogram = stft(signal, n_fft=512, hop_length=256)
    spectrogram = np.concatenate((spectrogram.real, spectrogram.imag), axis=0)

    # pad to multiple of 64
    spectrogram = np.pad(spectrogram, ((0, (max_feature_size - spectrogram.shape[0])), (0, (max_time_steps - spectrogram.shape[1]))), 'constant').transpose()
    # spectrogram = np.pad(spectrogram, ((0, 64-(spectrogram.shape[0]%64)), (0, 64-(spectrogram.shape[1]%64))), 'constant').transpose()
    print(spectrogram.shape)

    return spectrogram

def transform(inputs, target):
    # convert audio to spectrogram
    inputs = generate_spectrogram(inputs)
    inputs = np.expand_dims(inputs, axis=0)
    print(inputs.shape)

    target = np.array([emotions[target]])
    time_steps = inputs.shape[1]
    features_size = inputs.shape[2]
    time_block_count = time_steps // 64
    feature_block_count = features_size // 64

    # initializing the resulting spectrogram
    spectrogram = np.zeros((time_steps, features_size), dtype=np.float64)

    generator = build_generator(64, 64, time_block_count, feature_block_count)
    generator.load_weights("best_model.h5")

    for y in range(time_block_count):
        Y = np.repeat(y, 1).reshape(-1, 1)
        for x in range(feature_block_count):
            start_y = y * 64
            start_x = x * 64
            X = np.repeat(x, 1).reshape(-1, 1)

            # update the generator via the discriminator's error
            spectrogram[start_y:(start_y + 64), start_x:(start_x + 64)] = generator.predict([inputs[:, start_y:(start_y + 64), start_x:(start_x + 64)], Y, X, target], batch_size=1)

    return spectrogram
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Speech Input audio file to run the model on")
    parser.add_argument('target', help="One of the 8 target emotions (happy, sad, surprise, neutral, calm, angry, disgust, fear)")
    args = parser.parse_args()

    spectrogram = transform(args.input, args.target)
    reconstruct(spectrogram)

if __name__=='__main__':
    main()
