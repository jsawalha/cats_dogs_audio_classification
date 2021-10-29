import random, os
import numpy as np
import pandas as pd
import librosa.display
import librosa
import python_speech_features as psf
import matplotlib.pyplot as plt

#Set up the paths for both training and testing dogs/cats
path = r'/home/jeff/Documents/cats_dogs_audio_signal_processing/cats_dogs/'

train_cats = r'train/cat'
train_dogs = r'train/dog'

test_cats = r'test/cat'
test_dogs = r'test/dog'

#Build a function to plot samples of cats and or dogs
def plot_sound(path):
    plt.figure(figsize=(12,5))
    x, sr  = librosa.load(path)
    print(r'length - {len(x}; sample rate - {sr}')
    librosa.display.waveplot(x, sr = sr)
    return x

#Another function to reduce the length of signals based on SILENCE or near silence in the clips. Helps increase our signal
def envelope(y,rate,threshold):
    mask=[]
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10),min_periods=1,center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

#Get our Mel Spectrogram
def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

#Show Spectrogram
def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel',
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title(title)
    plt.axis('off')
    plt.show()

#Settings to get our mel spectrogram
class conf:
    sampling_rate = 22050
    duration = 1
    hop_length = 347*duration
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 60
    n_fft = n_mels * 20
    samples = sampling_rate * duration

#Here, print the number of files we have for each section
#Also, get longest length of sound to zero pad for our CNN
aud_names = []
len_auds = []
max_len = 395560 #longest file [number of frames] after masking and noise removal

for folder, subfolders, filenames in os.walk(path):
    for aud in filenames:
            if aud.endswith(".wav"):
                aud_names.append(folder + '/' + aud)
                x, sr = librosa.load(folder + '/' + aud)
                x = librosa.util.normalize(x)
                mask = envelope(x, sr, 0.005) #Here, make our mask to figure out our audio sound threshold
                x = x[mask] #overwrite the file with the mask included, to shorten the signal
                # zero_pad_len = max_len - len(x)
                # x = np.concatenate([x, np.zeros(zero_pad_len)])
                x_mel = audio_to_melspectrogram(conf, x)
                print(x_mel.shape)
                show_melspectrogram(conf, x_mel)
                plt.savefig(folder.split('/t')[0] + '/non_pad_MFCC_' + folder.split('cats_dogs/')[1] + '/' + aud.split('.')[0] + '.jpeg', bbox_inches='tight',
                            transparent=True, pad_inches=0.0)
                plt.close()
max_length, max_file = (max(len_auds), aud_names[len_auds.index(max(len_auds))])

