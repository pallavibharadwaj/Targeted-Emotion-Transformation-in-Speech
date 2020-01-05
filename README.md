# Targeted Emotion Transformation in Speech

Style transfer on images is a very active research topic and has succeeded with the usage of convolutional neural networks. Extending the concept to speech has been quite a challenge. It is difficult to transfer features such as emotion and accent.
We restricted our study to the emotional component of speech and trained a model that could transform the emotion contained in that speech sample to another.

### Dataset ###
We used the [RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) and [SAVEE](https://www.kaggle.com/barelydedicated/savee-database) databases for training the model developed. We trained on 8 kinds of emotions: neutral, calm, happy, sad, angry, fearful, surprise, and disgust. Each of these is represented by a label from 0-7.

### Conditional GAN ###
We used a conditional GAN to transform the emotion in a given speech sample to a given 'target' emotion, specified by the target label that is provided to the generator.
Along with the label for the target emotion, The generator is fed the input speech signal in the form of a spectrogram, generated using Short-Time Fourier Transform (STFT). The generator outputs a spectrogram for the input speech signal with the transformed emotional component.

### Discriminator Models ###
The discriminator objectives are:
- To correctly label real speech signals which are coming from the training dataset as ’real’.
- To correctly label generated images that are coming from the generator as ’fake’.

We experimented with three different neural network architectures for the discriminator:
- Temporal Convolutional Network(TCN)
- Long Short-Term Memory Network(LSTM)
- Convolutional Neural Network.

On training each of these model architectures for over 300 epochs, the loss was recorded to be the lowest for the convolutional network.

### How to Run ###

Required dependencies and packages:

```
conda env create -f environment.yml
source activate speech_emotion_transformation
```

Testing the model: (transformed speech - target.wav)
```
python model.py neutral_test_speech.wav happy
```
