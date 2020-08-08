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

input_text = ''
for text in pdf:
    input_text += text

# Synthesize with melgan (alpha=1.0)
s = time()
wav = synthesize(input_text, tts_model, voc_melgan, alpha=1)
write('sample.wav', hp.sample_rate, wav)
e = time()
print(f"Audio conversion at {len(input_text)//(e-s)} words per second")
print('Audio conversion successful')
print("File saved as sample.wav")

song = AudioSegment.from_wav("sample.wav")
play(song)

