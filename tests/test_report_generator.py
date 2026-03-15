import os
import json
import pytest
from scripts.generate_real_markdown_report import generate_graphs_and_report

def test_generate_report(tmp_path):
    out_dir = tmp_path / "metrics"
    out_dir.mkdir()
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir()
    
    # Create mock JSON inputs
    hybrid_json = out_dir / "domain_metrics.json"
    baseline_json = out_dir / "real_baseline_metrics.json"
    
    with open(hybrid_json, "w") as f:
        json.dump({"VSD": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}}, f)
        
    with open(baseline_json, "w") as f:
        json.dump({"VSD": {
            "YAMNet": {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5},
            "PANNs": {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1": 0.6},
            "VGGish": {"accuracy": 0.7, "precision": 0.7, "recall": 0.7, "f1": 0.7}
        }}, f)
                           
    markdown_path = out_dir / "final_report.md"
    generate_graphs_and_report(str(hybrid_json), str(baseline_json), str(graphs_dir), str(markdown_path))
    
    assert markdown_path.exists()
    assert (graphs_dir / "VSD_accuracy.png").exists()
    assert (graphs_dir / "VSD_f1.png").exists()
