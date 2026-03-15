import os
import sys
import importlib.util
import pytest

# Dynamically import the script from the hidden .local_tools directory
script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".local_tools", "run_nlp_benchmarks.py"))

def load_module():
    spec = importlib.util.spec_from_file_location("run_nlp", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_nlp"] = module
    spec.loader.exec_module(module)
    return module

def test_mock_evaluation_returns_all_architectures():
    # Attempting to load a module that doesn't exist yet will fail (RED phase)
    # or calling a function that isn't implemented will fail.
    run_nlp = load_module()
    
    results = run_nlp.mock_evaluation()
    
    expected_architectures = [
        "BERT (Baseline)",
        "DistilBERT (Edge)",
        "DeBERTa (SOTA)",
        "ALBERT (Lightweight)",
        "RoBERTa (Current)"
    ]
    
    for arch in expected_architectures:
        assert arch in results, f"Missing architecture: {arch}"
        
        metrics = results[arch]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        assert isinstance(metrics["accuracy"], float)
