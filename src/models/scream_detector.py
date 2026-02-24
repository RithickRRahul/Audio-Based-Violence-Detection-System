import numpy as np
import librosa
import re

class ScreamDetector:
    """
    Acts as an acoustic and text-based fail-safe for violence detection.
    If people are screaming or grunting, Whisper may return '' or hallmark repetitive characters.
    This detector provides boolean overrides to catch what the semantic NLP models miss.
    """
    
    def __init__(self):
        # Text-based scream patterns (Whisper occasionally tries to transliterate screams)
        self.scream_regex = re.compile(r'(ah+)|(uh+)|(oh+)|(eh+)|(ee+)', re.IGNORECASE)
        self.distress_words = ['help', 'stop', 'no', 'police', 'quit', 'ow', 'ouch']
        
    def detect(self, transcript: str) -> bool:
        """
        Detects if a transcript contains obvious scream transliterations 
        or bare distress words that Whisper couldn't form into sentences.
        """
        if not transcript or transcript.strip() == "":
            return False
            
        text = transcript.lower()
        
        # Check for scream patterns (e.g. "ahhhhh", "ohhh")
        if self.scream_regex.search(text):
            return True
            
        # Check for isolated distress words
        words = text.split()
        if len(words) <= 3:
            for w in words:
                if w in self.distress_words:
                    return True
                    
        return False
        
    def detect_acoustic(self, audio_segment: np.ndarray, sr: int) -> bool:
        """
        Detects acoustic screams using spectral analysis.
        Screams typically have very high energy in the 1kHz - 3kHz range
        and exhibit harsh, broadband spectral characteristics.
        """
        # Feature 1: Spectral Centroid (screams have high centroid)
        centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0]
        mean_centroid = np.mean(centroid)
        
        # Feature 2: Spectral Rolloff (screams carry energy in high frequencies)
        rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr, roll_percent=0.85)[0]
        mean_rolloff = np.mean(rolloff)
        
        # Feature 3: Zero Crossing Rate (harshness/noise-like quality of screams)
        zcr = librosa.feature.zero_crossing_rate(audio_segment)[0]
        mean_zcr = np.mean(zcr)
        
        # Empirically defined thresholds for human screaming vs background noise
        # These are basic heuristics acting as a fallback for the CMAG network
        is_high_pitch = mean_centroid > 2500 and mean_rolloff > 4000
        is_harsh = mean_zcr > 0.15
        
        # Additionally check energy to ignore faint high-pitch distant noises
        rms = librosa.feature.rms(y=audio_segment)[0]
        mean_rms = np.mean(rms)
        is_loud = mean_rms > 0.05
        
        return bool(is_high_pitch and is_harsh and is_loud)

    def detect_impact(self, audio_segment: np.ndarray, sr: int) -> bool:
        """
        Detects sudden loud transient impacts like punches, slaps, or gunshots.
        Works by analyzing the sudden jump in RMS energy (a spike).
        """
        # Calculate RMS energy frame-by-frame
        rms = librosa.feature.rms(y=audio_segment)[0]
        
        # We are looking for a sudden, massive jump in volume from one frame to the next
        # typical of a percussive hit.
        if len(rms) < 2:
            return False
            
        # Calculate the delta (change) in energy between consecutive frames
        energy_deltas = np.diff(rms)
        
        # If there is a massive positive spike (> 0.08 jump in a single frame)
        # And the overall RMS peak is loud enough (> 0.1)
        max_delta = np.max(energy_deltas)
        max_rms = np.max(rms)
        
        is_impact = (max_delta > 0.05) and (max_rms > 0.08)
        
        return bool(is_impact)
