import pytest
import numpy as np
import os
import soundfile as sf
from src.data.audio_utils import load_audio, segment_audio, extract_mel_spectrogram

@pytest.fixture
def dummy_audio_file(tmp_path):
    """Creates a temporary 1-second 44.1kHz stereo audio file."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Create stereo signal (sine waves)
    y_left = np.sin(2 * np.pi * 440 * t)
    y_right = np.sin(2 * np.pi * 880 * t)
    y_stereo = np.vstack((y_left, y_right)).T
    
    file_path = tmp_path / "dummy_stereo.wav"
    sf.write(str(file_path), y_stereo, sr)
    return str(file_path)

def test_load_audio_converts_to_mono_and_resamples(dummy_audio_file):
    """Test that load_audio correctly resamples to 16kHz and converts to mono."""
    target_sr = 16000
    y, sr = load_audio(dummy_audio_file, target_sr=target_sr)
    
    assert sr == target_sr
    assert len(y.shape) == 1
    assert len(y) == target_sr

def test_segment_audio_short_clip_is_padded():
    """Test that a short audio clip is padded to the full segment length."""
    sr = 16000
    seg_len = 4.0
    target_samples = int(sr * seg_len) # 64000
    
    # Create a 2-second clip (too short)
    y_short = np.ones(sr * 2)
    
    segments = segment_audio(y_short, sr, seg_len=seg_len)
    
    assert len(segments) == 1
    assert len(segments[0]) == target_samples
    assert np.all(segments[0][sr * 2:] == 0) # Padded with zeros

def test_segment_audio_long_clip_is_split():
    """Test that a long audio clip is split into multiple segments."""
    sr = 16000
    seg_len = 4.0
    
    # Create a 10-second clip
    y_long = np.ones(sr * 10)
    
    segments = segment_audio(y_long, sr, seg_len=seg_len)
    
    # 10s / 4s = 2 full segments + 1 padded segment = 3 segments total
    assert len(segments) == 3
    assert len(segments[0]) == sr * 4
    assert len(segments[1]) == sr * 4
    assert len(segments[2]) == sr * 4  # The 2s remainder is padded to 4s

def test_extract_mel_spectrogram_shape():
    """Test that the mel spectrogram output shape is correct for a CNN."""
    sr = 16000
    seg_len = 4.0
    y = np.random.randn(int(sr * seg_len)) # 1 full segment
    
    n_mels = 128
    hop_length = 512
    # Expected time steps = 64000 / 512 + 1 = 126
    
    mel_spec = extract_mel_spectrogram(y, sr, n_mels=n_mels, n_fft=2048, hop_length=hop_length)
    
    assert len(mel_spec.shape) == 2
    assert mel_spec.shape[0] == n_mels
    assert mel_spec.shape[1] == 126
