import os
import json
import pytest
from scripts.evaluate_domain_specific import evaluate_individual_datasets

def test_evaluate_individual_datasets_saves_json(tmp_path):
    output_file = tmp_path / "domain_metrics.json"
    
    # Run evaluation with test_run=True (very small subset or mocked model)
    evaluate_individual_datasets(output_path=str(output_file), test_run=True)
    
    assert output_file.exists()
    
    with open(output_file, "r") as f:
        data = json.load(f)
        
    # Check that datasets are present
    for dataset in ["VSD", "CREMA-D", "ESC-50", "UrbanSound8K"]:
        assert dataset in data
        assert "accuracy" in data[dataset]
        assert "precision" in data[dataset]
        assert "recall" in data[dataset]
        assert "f1" in data[dataset]
        
    md_file = tmp_path / "domain_specific_evaluation.md"
    assert md_file.exists()
