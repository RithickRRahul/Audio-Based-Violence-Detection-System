import os
import json
import pytest
from scripts.evaluate_actual_baselines import run_real_baselines

def test_run_real_baselines_json_output(tmp_path):
    output_file = tmp_path / "baseline_metrics.json"
    
    # Run evaluation with test_run=True (very small subset or mocked model)
    # This should not download the huge models or run for hours during Pytest
    run_real_baselines(output_path=str(output_file), test_run=True)
    
    assert output_file.exists()
    
    with open(output_file, "r") as f:
        data = json.load(f)
        
    for dataset in ["VSD", "CREMA-D"]:
        assert dataset in data
        assert "YAMNet" in data[dataset]
        assert "VGGish" in data[dataset]
        assert "PANNs" in data[dataset]
        
        for model in ["YAMNet", "VGGish", "PANNs"]:
            assert "accuracy" in data[dataset][model]
            assert "precision" in data[dataset][model]
            assert "recall" in data[dataset][model]
            assert "f1" in data[dataset][model]
