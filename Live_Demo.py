# Load pretrained models
from notebook_utils.synthesize import (
    get_forward_model, get_melgan_model, get_wavernn_model, synthesize, init_hparams)
from utils import hparams as hp
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.playback import play
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pdftotext
from time import time
import numpy as np
from pydub import AudioSegment

# Checking if GPU available
# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

init_hparams('notebook_utils/pretrained_hparams.py')
tts_model = get_forward_model('pretrained/forward_400K.pyt')
voc_melgan = get_melgan_model()
voc_wavernn = get_wavernn_model('pretrained/wave_575K.pyt')

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filelocation = askopenfilename() # open the dialog GUI

with open(filelocation, "rb") as f:  # open the file in reading (rb) mode and call it f
    pdf = pdftotext.PDF(f)  # store a text version of the pdf file f in pdf variable

# print(len(str(pdf).split()))
num_words = 0
initial = "Welcome to audio file of pdf by PEC ACM."
wav = synthesize(initial, tts_model, voc_melgan, alpha=1)

pgno = 0
s = time()
for text in pdf:
    pgno+=1
    l = str(text).split()
    c = 0
    num_words+=len(l)
    for i in range(0, len(l), 10):
        temp = ''
        for j in range(i, min(len(l), i+10)):
            temp+=l[j]
            temp+=' '
        wav1 = synthesize(temp, tts_model, voc_melgan, alpha=1)
        wav = np.append(wav, wav1)
    print("Page no: ",pgno, "completed!!")


write('sample.wav', hp.sample_rate, wav)
e = time()

# print(num_words, e, s)
print(f"Audio conversion at {num_words//(e-s)} words per second")
# print('Audio conversion successful')

AudioSegment.from_wav("sample.wav").export("Generated-Audio.mp3", format="mp3")
print("Audio file saved as Generated-Audio.mp3")

# song = AudioSegment.from_wav("sample.wav")
# play(song)

