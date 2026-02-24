import os
import torch
import numpy as np
import whisper
from src.config import (
    SAMPLE_RATE, SEGMENT_LENGTH, N_MELS, N_FFT, HOP_LENGTH,
    SAVED_MODELS_DIR
)
from src.data.audio_utils import load_audio, segment_audio, extract_mel_spectrogram
from src.models.audio_encoder import AudioEncoder
from src.models.nlp_encoder import TextEncoder
from src.models.scream_detector import ScreamDetector
from src.models.cmag_v2 import EnhancedCMAG
from src.data.fast_vad import FastVAD
from src.models.temporal_tracker import RuleBasedTemporalTracker
from src.utils.logger import get_logger

logger = get_logger("inference_pipeline")


class ViolenceDetectionPipeline:
    """
    Full end-to-end inference pipeline for violence detection.
    
    Flow:
    1. Load audio file → resample to 16kHz mono
    2. Segment into 4-second chunks
    3. For each segment:
       a. FastVAD check: Skip STT if no speech
       b. Extract mel-spectrogram → AudioEncoder → audio_emb
       c. (If speech) Whisper STT → TextEncoder → text_emb
       d. ScreamDetector → distress signals
       e. CMAG-v2 / Adaptive fusion → segment_score
    4. RuleBasedTemporalTracker → Trend & escalation score
    5. Return: per-segment results + temporal score + alert status
    """
    
    def __init__(self, device: str = None, whisper_model: str = "tiny"):
        """
        Initialize the pipeline with all model components.
        
        Args:
            device: 'cuda' or 'cpu'. Auto-detects if None.
            whisper_model: Whisper model size ('tiny', 'base', 'small')
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing on {self.device}...", extra={"extra_info": {"device": str(self.device)}})
        
        # Load all model components
        self.audio_encoder = AudioEncoder().to(self.device).eval()
        self.text_encoder = TextEncoder()
        
        # Threat Detectors
        self.scream_detector = ScreamDetector()
        self.vad = FastVAD()
        
        # Fusion Models
        self.cmag = EnhancedCMAG().to(self.device).eval()
        
        # Temporal analysis (Replaced Bi-LSTM with Deterministic Rules)
        self.temporal = RuleBasedTemporalTracker(window_size=5)
        
        # Load Whisper for speech-to-text
        logger.info(f"Loading Whisper ({whisper_model})...", extra={"extra_info": {"model": whisper_model}})
        self.whisper_model = whisper.load_model(whisper_model, device=str(self.device))
        
        logger.info("Pipeline Ready.")
    
    def load_weights(self, models_dir: str = SAVED_MODELS_DIR):
        """Load trained model weights from disk."""
        audio_path = os.path.join(models_dir, "audio_encoder.pth")
        cmag_path = os.path.join(models_dir, "cmag_v2.pth")
        
        if os.path.exists(audio_path):
            self.audio_encoder.load_state_dict(torch.load(audio_path, map_location=self.device))
            logger.info(f"Loaded audio encoder from {audio_path}", extra={"extra_info": {"path": audio_path}})
        
        if os.path.exists(cmag_path):
            self.cmag.load_state_dict(torch.load(cmag_path, map_location=self.device))
            logger.info(f"Loaded CMAG-v2 from {cmag_path}", extra={"extra_info": {"path": cmag_path}})
    
    def _transcribe_segment(self, audio_segment: np.ndarray) -> str:
        """Transcribe a single audio segment using Whisper."""
        try:
            # Whisper expects float32 audio at 16kHz
            audio_float = audio_segment.astype(np.float32)
            # Pad or trim to 30 seconds (Whisper requirement)
            audio_padded = whisper.pad_or_trim(audio_float)
            mel = whisper.log_mel_spectrogram(audio_padded).to(self.device)
            options = whisper.DecodingOptions(language="en", fp16=(self.device.type == "cuda"))
            result = whisper.decode(self.whisper_model, mel, options)
            
            # Reject high probability of no speech (where Whisper tends to hallucinate)
            if hasattr(result, "no_speech_prob") and result.no_speech_prob > 0.6:
                return ""
                
            return result.text.strip()
        except Exception as e:
            logger.error(f"Whisper transcription error: {str(e)}", extra={"extra_info": {"error": str(e)}})
            return ""
    
    @torch.no_grad()
    def process_file(self, file_path: str) -> dict:
        """
        Process a single audio file through the full pipeline.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            dict with:
            - 'file_path': input path
            - 'segments': list of per-segment results
            - 'temporal_score': file-level violence probability
            - 'final_state': 'VIOLENCE' or 'SAFE'
        """
        # Step 1: Load and segment
        y, sr = load_audio(file_path, target_sr=SAMPLE_RATE)
        segments = segment_audio(y, sr, seg_len=SEGMENT_LENGTH)
        
        segment_results = []
        
        # Reset the temporal tracker for a new file
        self.temporal = RuleBasedTemporalTracker(window_size=5)
        
        # We need this to ensure the final score tracks the max violation
        file_max_score = 0.0
        
        for i, seg in enumerate(segments):
            start_time = i * SEGMENT_LENGTH
            end_time = start_time + SEGMENT_LENGTH
            
            # Step 2a: Audio branch
            mel_spec = extract_mel_spectrogram(seg, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                audio_emb = self.audio_encoder(mel_tensor)
                
            # Step 2b: FastVAD Check
            has_speech, vad_details = self.vad.has_speech(seg, sr)
            
            transcript = ""
            threat_score = 0.0
            
            # Step 2c: Text branch (Conditional on Speech)
            if has_speech:
                transcript = self._transcribe_segment(seg)
                
                # Filter out hallucinated punctuation & micro words like "." or "you"
                import string
                cleaned = transcript.translate(str.maketrans('', '', string.punctuation)).strip()
                
                if transcript and len(cleaned) >= 4:
                    text_emb = self.text_encoder.get_embeddings([transcript]).to(self.device)
                    threat_score = self.text_encoder.get_threat_score(transcript)
                else:
                    has_speech = False
                    transcript = transcript if transcript else ""
                    
            if not has_speech:
                # Bypass Whisper STT to save compute, use a zero-vector for CMAG fusion
                text_emb = torch.zeros((1, 768), device=self.device)
                threat_score = 0.0
            
            # Step 2d: Scream detection
            scream_text = self.scream_detector.detect(transcript) if transcript else False
            scream_acoustic = self.scream_detector.detect_acoustic(seg, sr)
            is_impact = self.scream_detector.detect_impact(seg, sr)
            
            # Step 2e: Adaptive Fusion
            # We ALWAYS route to CMAG because CMAG holds the violence classification head.
            # If `has_speech` is false, it fuses the audio with the zero-vector, ensuring 
            # correct mathematical distribution.
            with torch.no_grad():
                seg_score = self.cmag(audio_emb, text_emb, return_features=False)
            baseline_score = seg_score.item()
                
            # Step 3: Temporal Tracking (Deterministic)
            effective_score = baseline_score
            if scream_acoustic or scream_text or threat_score > 0.6:
                effective_score = max(effective_score, 0.85)
            elif is_impact:
                # Add a proportional boost for impact transients instead of flatlining
                effective_score = min(effective_score + 0.25, 0.95)
                
            temporal_analysis = self.temporal.update(effective_score)
            escalation_score = temporal_analysis["escalation_score"]
            
            file_max_score = max(file_max_score, escalation_score, effective_score)
            is_violent = escalation_score > 0.5
            
            segment_results.append({
                "segment_index": i,
                "timestamp": f"{start_time:.1f}s - {end_time:.1f}s",
                "vad_speech": has_speech,
                "audio_score": baseline_score,
                "nlp_score": threat_score,
                "transcript": transcript,
                "scream_text": scream_text,
                "scream_acoustic": scream_acoustic,
                "temporal_trend": temporal_analysis["trend"],
                "temporal_segment_score": escalation_score,
                "state": "VIOLENCE" if is_violent else "SAFE"
            })
            
        return {
            "file_path": file_path,
            "segments": segment_results,
            "temporal_score": file_max_score,
            "final_state": "VIOLENCE" if file_max_score > 0.5 else "SAFE"
        }
