import pytest
import numpy as np
import librosa
from src.data.fast_vad import FastVAD

@pytest.fixture
def empty_audio():
    # 1 second of perfect silence at 16kHz
    return np.zeros(16000, dtype=np.float32)

@pytest.fixture
def loud_noise_audio():
    # 1 second of white noise (e.g. wind/static)
    np.random.seed(42)
    return np.random.normal(0, 0.5, 16000).astype(np.float32)

@pytest.fixture
def speech_audio():
    # We'll mock a sine wave in human vocal range (~300Hz) wrapped in an envelope
    t = np.linspace(0, 1, 16000)
    wave = 0.5 * np.sin(2 * np.pi * 300 * t) 
    # Add some harmonics typical of speech
    wave += 0.2 * np.sin(2 * np.pi * 600 * t)
    return wave.astype(np.float32)

def test_vad_detects_silence(empty_audio):
    vad = FastVAD()
    has_speech, details = vad.has_speech(empty_audio, sr=16000)
    assert has_speech is False
    assert details["reason"] == "too_quiet"

def test_vad_detects_pure_noise_as_non_speech(loud_noise_audio):
    vad = FastVAD()
    has_speech, details = vad.has_speech(loud_noise_audio, sr=16000)
    # White noise has very high zero-crossing rate and centroid, 
    # well outside human speech bounds
    assert has_speech is False
    assert details["reason"] in ["too_noisy", "too_harsh_for_speech"]

def test_vad_detects_human_speech_frequencies(speech_audio):
    vad = FastVAD()
    has_speech, details = vad.has_speech(speech_audio, sr=16000)
    # The clean 300Hz+600Hz tone should be firmly in the human speech band
    assert has_speech is True
    assert details["reason"] == "speech_detected"
