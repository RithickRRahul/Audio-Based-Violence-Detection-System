import os
import json
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.inference.pipeline import ViolenceDetectionPipeline
from src.data.datasets import load_vsd_dataset, load_cremad_dataset, load_esc50_dataset, load_urbansound_dataset

def process_dataset(config: dict, files, labels, pipeline):
    predictions = []
    true_labels = []
    
    for fpath, label in tqdm(zip(files, labels), total=len(files), desc="Processing"):
        res = pipeline.process_file(fpath, ablation_config=config)
        pred = 1 if res["final_state"] == "VIOLENCE" else 0
        predictions.append(pred)
        true_labels.append(label)
        
    acc = float(accuracy_score(true_labels, predictions))
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }

def run_study(output_path: str = "docs/performance_metrics/ablation_results.json", test_run: bool = False):
    import warnings
    warnings.filterwarnings("ignore")
    
    print("Loading test dataset...")
    all_files, all_labels = [], []
    for loader in [load_vsd_dataset, load_cremad_dataset, load_esc50_dataset, load_urbansound_dataset]:
        f, l = loader()
        all_files.extend(f)
        all_labels.extend(l)
        
    combined = list(zip(all_files, all_labels))
    random.seed(42)
    random.shuffle(combined)
    
    # 200 samples gives a confident and relatively quick evaluation
    sample_size = 10 if test_run else 200
    subset = combined[:sample_size]
    files = [x[0] for x in subset]
    labels = [x[1] for x in subset]
    
    configs = {
        "Baseline": {},
        "No_Temporal": {"use_temporal": False},
        "No_CMAG": {"use_cmag": False},
        "No_Audio": {"use_audio": False},
        "No_Text": {"use_nlp": False},
        "No_Failsafes": {"use_scream": False},
        "No_VAD": {"use_vad": False}
    }
    
    print(f"Initializing pipeline...")
    pipeline = ViolenceDetectionPipeline(whisper_model="tiny")
    pipeline.load_weights()
    
    results = {}
    
    for name, config in configs.items():
        print(f"\n================ Eval: {name} ================")
        metrics = process_dataset(config, files, labels, pipeline)
        results[name] = metrics
        print(f"Metrics: {metrics}")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Markdown Report
    md_path = os.path.join(os.path.dirname(output_path), "ablation_report.md")
    md = "# CMAG-v2 End-to-End Ablation Study (Pure BERT Retraining)\n\n"
    md += "Performance of the system when disabling specific sub-modules via inference configuration flags.\n\n"
    for name, m in results.items():
        md += f"## Configuration: {name}\n"
        md += f"- **Accuracy:** {m['accuracy']:.4f}\n"
        md += f"- **Precision:** {m['precision']:.4f}\n"
        md += f"- **Recall:** {m['recall']:.4f}\n"
        md += f"- **F1-Score:** {m['f1']:.4f}\n\n"
        
    with open(md_path, "w") as f:
        f.write(md)
        
    print(f"\nSaved metrics: {output_path}")
    print(f"Saved report: {md_path}")

if __name__ == "__main__":
    run_study()
