import numpy as np
import librosa

class FastVAD:
    """
    A lightweight, rule-based Voice Activity Detector (VAD).
    
    Instead of running a heavy neural network to detect speech (which YAMNet did using silero-vad),
    we use deterministic acoustic heuristics:
    1. Energy (RMS): Is there sound at all?
    2. Zero-Crossing Rate (ZCR): Is it tonal (speech) or purely noisy (static/hiss)?
    3. Spectral Centroid: Is the bulk of the energy located in the human voice band (300Hz - 3400Hz)?
    """
    
    def __init__(self, min_energy=0.01, min_speech_zcr=0.01, max_speech_zcr=0.15, min_centroid=200, max_centroid=4000):
        # Very quiet sounds are ignored
        self.min_energy = min_energy
        
        # Speech has tonal vowels (low ZCR) and fricative consonants (higher ZCR). 
        # White noise has extremely high ZCR (>0.4).
        self.min_speech_zcr = min_speech_zcr
        self.max_speech_zcr = max_speech_zcr
        
        # Human speech energy averages around 200Hz to 4000Hz.
        self.min_centroid = min_centroid
        self.max_centroid = max_centroid

    def has_speech(self, audio_segment: np.ndarray, sr: int) -> tuple[bool, dict]:
        """
        Determines if an audio segment contains likely human speech.
        
        Args:
            audio_segment: 1D numpy array of audio samples.
            sr: Sample rate (e.g., 16000).
            
        Returns:
            (has_speech: bool, details: dict)
        """
        if len(audio_segment) == 0:
            return False, {"reason": "empty_array"}
            
        # 1. Energy Check (RMS)
        rms = librosa.feature.rms(y=audio_segment)[0]
        mean_rms = np.mean(rms)
        if mean_rms < self.min_energy:
            return False, {"reason": "too_quiet", "rms": mean_rms}
            
        # 2. Tonal/Noise Check (Zero-Crossing Rate)
        zcr = librosa.feature.zero_crossing_rate(y=audio_segment)[0]
        mean_zcr = np.mean(zcr)
        if mean_zcr > self.max_speech_zcr:
            return False, {"reason": "too_noisy", "zcr": mean_zcr}
        if mean_zcr < self.min_speech_zcr:
            return False, {"reason": "too_tonal", "zcr": mean_zcr}
            
        # 3. Frequency Band Check (Spectral Centroid)
        centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0]
        mean_centroid = np.mean(centroid)
        if mean_centroid > self.max_centroid:
            return False, {"reason": "too_harsh_for_speech", "centroid": mean_centroid}
        if mean_centroid < self.min_centroid:
            return False, {"reason": "too_low_for_speech", "centroid": mean_centroid}
            
        return True, {
            "reason": "speech_detected", 
            "rms": mean_rms, 
            "zcr": mean_zcr, 
            "centroid": mean_centroid
        }
