import pytest
import numpy as np
import os
import soundfile as sf
from src.inference.pipeline import ViolenceDetectionPipeline

from unittest.mock import patch, MagicMock

@pytest.fixture
def dummy_pipeline():
    # Initialize the pipeline without loading any real weights to avoid overhead
    # We mock whisper.load_model so it doesn't try to download or load a real model into RAM
    with patch('whisper.load_model') as mock_whisper:
        mock_whisper.return_value = MagicMock()
        pipeline = ViolenceDetectionPipeline(device="cpu", whisper_model="tiny")
    return pipeline

@pytest.fixture
def silent_audio_file(tmp_path):
    file_path = tmp_path / "silent.wav"
    # 4 seconds of silence at 16kHz
    y = np.zeros(16000 * 4, dtype=np.float32)
    sf.write(file_path, y, 16000)
    return str(file_path)

def test_pipeline_skips_nlp_on_silence(dummy_pipeline, silent_audio_file):
    # Process a completely silent file
    results = dummy_pipeline.process_file(silent_audio_file)
    
    assert len(results["segments"]) == 1
    seg = results["segments"][0]
    
    # FastVAD should have detected no speech
    assert seg["vad_speech"] is False
    
    # Transcript should be empty because we skipped STT
    assert seg["transcript"] == ""
    
    # NLP Threat score should be exactly 0.0
    assert seg["nlp_score"] == 0.0
    
    # Audio score should be the fallback baseline score (gated)
    assert isinstance(seg["audio_score"], float)
    
    # Since it's silence, it should not trigger violence
    assert seg["state"] == "SAFE"

def test_pipeline_end_to_end_routing(dummy_pipeline):
    # This just ensures the pipeline can instantiate and load weights
    # without any Shape mismatch errors throwing exceptions.
    try:
        dummy_pipeline.load_weights()
        assert True
    except Exception as e:
        pytest.fail(f"load_weights failed with error: {e}")
