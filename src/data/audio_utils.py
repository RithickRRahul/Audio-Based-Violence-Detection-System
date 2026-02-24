import librosa
import numpy as np

def load_audio(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Loads an audio file, converts it to mono, and resamples it.
    
    Args:
        file_path: Path to the audio file.
        target_sr: Desired sample rate (default 16000Hz).
        
    Returns:
        tuple containing the audio time series (1D numpy array) and the sample rate.
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return y, sr

def segment_audio(y: np.ndarray, sr: int, seg_len: float = 4.0) -> list[np.ndarray]:
    """
    Splits an audio array into fixed-length segments, padding the last one if necessary.
    
    Args:
        y: Audio time series.
        sr: Sample rate.
        seg_len: Segment length in seconds.
        
    Returns:
        A list of audio segments, each of exact length `sr * seg_len`.
    """
    samples_per_seg = int(sr * seg_len)
    num_full_segments = len(y) // samples_per_seg
    segments = []
    
    for i in range(num_full_segments):
        start = i * samples_per_seg
        segments.append(y[start : start + samples_per_seg])
        
    remainder = len(y) % samples_per_seg
    if remainder > 0 or len(y) == 0:
        start = num_full_segments * samples_per_seg
        last_segment = y[start:]
        # Pad with zeros
        padded = np.pad(last_segment, (0, samples_per_seg - len(last_segment)), mode='constant')
        segments.append(padded)
        
    return segments

def extract_mel_spectrogram(y: np.ndarray, sr: int, n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Extracts a mel-spectrogram from an audio signal.
    
    Args:
        y: Audio time series.
        sr: Sample rate.
        n_mels: Number of Mel bands to generate.
        n_fft: Length of the FFT window.
        hop_length: Number of samples between successive frames.
        
    Returns:
        2D numpy array containing the mel-spectrogram in decibels.
    """
    # Extract Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    # Convert to log scale (decibels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db
