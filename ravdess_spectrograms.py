import numpy as np
import librosa
import os
import matplotlib.pyplot as plt

input_folder = 'ravdess-emotional-speech-audio'
output_folder = 'spectrograms/ravdess'

for actor_folder in os.listdir(input_folder):
   if os.path.isdir(os.path.join(input_folder, actor_folder)):
       input_path = os.path.join(input_folder, actor_folder)
       output_path = os.path.join(output_folder, actor_folder)
       if not os.path.isdir(output_path):
           os.mkdir(output_path)
       for input_file in os.listdir(input_path):
           if input_file.endswith('.wav'):
               print(os.path.join(input_path, input_file))
               signal, sr = librosa.load(os.path.join(input_path, input_file), sr =44100)
               
               spectrogram = librosa.stft(signal, n_fft=512)
               magnitude = np.abs(spectrogram)
               phase = np.angle(spectrogram)
               
               np.savetxt(os.path.join(output_path, input_file.split('.')[0] + '_mag.txt'), magnitude, fmt='%.6f')
               np.savetxt(os.path.join(output_path, input_file.split('.')[0] + '_phase.txt'), phase, fmt='%.6f')
                
# signal, sr = librosa.load(os.path.join('ravdess-emotional-speech-audio/Actor_01', '03-01-01-01-01-01-01.wav'))           
# spectrogram = librosa.stft(signal, n_fft=512)
# magnitude = np.abs(spectrogram)
# phase = np.angle(spectrogram)
# np.savetxt(os.path.join('spectrograms/ravdess', '03-01-01-01-01-01-01' + '_mag.txt'), magnitude, fmt='%.6f')
# np.savetxt(os.path.join('spectrograms/ravdess', '03-01-01-01-01-01-01' + '_phase.txt'), phase, fmt='%.6f')
# recon = librosa.istft(spectrogram)

# plt.figure(1)
# plt.plot(signal)
# plt.savefig('Signal.png')

# plt.figure(2)
# plt.plot(recon)
# plt.savefig('Reconstruction.png')
                
                
