import sounddevice as sd
import numpy as np

def record_audio(duration=5, fs=16000):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = audio.flatten()
    return audio, fs
