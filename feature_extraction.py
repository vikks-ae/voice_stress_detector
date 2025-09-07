import librosa
import numpy as np

def extract_features(audio, fs):
    # Pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=fs)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    # Volume (RMS energy)
    rms = librosa.feature.rms(y=audio)
    volume = np.mean(rms)
    
    # Speech speed (zero crossing rate as proxy for speed/articulation)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    speed = np.mean(zcr)
    
    return np.array([pitch, volume, speed])
