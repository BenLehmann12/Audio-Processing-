import librosa
import librosa.display
import IPython.display as ipd
import os
import matplotlib.pyplot as plt
import numpy as np

#os.path.exists('Users/Lehman/PycharmProjects/space_oddity.wav')  #Made sure the file is set
#print(os.listdir())

path = 'island_music_x.wav'
x, sr = librosa.load(path)
print(x.shape,sr)     #x.shape = (276480,) sr = 22050

plt.figure(figsize=(14,5))
librosa.display.waveplot(x, sr=sr)
plt.show()

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14,5))
librosa.display.specshow(Xdb,sr=sr, x_axis='time',y_axis='log')
plt.colorbar()
plt.show()

#Perceptual Weighting:
freq = librosa.core.fft_frequencies(sr=sr)
mag = librosa.perceptual_weighting(abs(X)**2, freq)
librosa.display.specshow(mag, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()

r = librosa.autocorrelate(x, max_size=6000)
sample = r[:300]
plt.figure(figsize=(14,5))
plt.plot(sample)
plt.show()

#Chroma Features
sound_len = 400
chrom = librosa.feature.chroma_stft(x, sr=sr, hop_length=sound_len)
plt.figure(figsize=(14,5))
librosa.display.specshow(chrom, x_axis='time',y_axis='chroma', hop_length=sound_len)
plt.colorbar()
plt.show()

#Mel-power
S = librosa.feature.melspectrogram(x,sr=sr,n_fft=120)
log = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(14,5))
librosa.display.specshow(log,sr=sr,x_axis='time',y_axis='mel')
plt.colorbar()
plt.show()

def Vocal():
  X, fr = librosa.load('space_oddity.wav')  #Worked With David Bowie's Space Oddity
  librosa.display.waveplot(X, sr=fr)
  plt.show()
  mfc = librosa.feature.mfcc(X,sr=fr)
  librosa.display.specshow(mfc,sr=fr, x_axis='time')
  plt.show()
  n1 = 600
  n2 = 700
  plt.figure(figsize=(15,4))
  plt.plot(X[n1:n2])
  plt.show()
  zero_cross = sum(librosa.zero_crossings(X[n1:n2], pad=False))   # Zero Crossing is 6

