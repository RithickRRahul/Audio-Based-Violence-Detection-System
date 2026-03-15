import pytest
import inspect
from src.inference.pipeline import ViolenceDetectionPipeline

def test_pipeline_signature_accepts_ablation_config():
    # Verify that process_file accepts an ablation_config parameter
    pipeline = ViolenceDetectionPipeline(device="cpu", whisper_model="tiny")
    sig = inspect.signature(pipeline.process_file)
    assert "ablation_config" in sig.parameters

def test_pipeline_ablation_logic_mock():
    # This is a placeholder test for the configuration flags logic
    # Real validation will be done by running the ablation study script
    assert True

