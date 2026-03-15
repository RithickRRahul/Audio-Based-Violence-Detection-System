"""
Proper Full-Pipeline CMAG Evaluation Script.

This script evaluates the complete CMAG violence detection pipeline
end-to-end on raw audio files using real Whisper STT, real BERT embeddings,
and the actual CMAG fusion gate — exactly as it would run in production.

It processes each test audio file through ViolenceDetectionPipeline and
computes sklearn classification metrics against known ground truth labels.
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    matthews_corrcoef
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

from src.data.datasets import (
    load_vsd_dataset, load_cremad_dataset,
    load_esc50_dataset, load_urbansound_dataset
)
from src.inference.pipeline import ViolenceDetectionPipeline


def get_test_files():
    """
    Load all 4 datasets and split using the same seed (42)
    as build_combined_dataset to get the identical test set.
    """
    all_files, all_labels = [], []
    
    for name, loader in [
        ("VSD", load_vsd_dataset),
        ("CREMA-D", load_cremad_dataset),
        ("ESC-50", load_esc50_dataset),
        ("UrbanSound8K", load_urbansound_dataset)
    ]:
        files, labels = loader()
        print(f"  {name}: {len(files)} files ({sum(labels)} violent, {len(labels) - sum(labels)} safe)")
        all_files.extend(files)
        all_labels.extend(labels)
    
    print(f"  Total: {len(all_files)} files")
    
    # Use the exact same split as training
    _, test_files, _, test_labels = train_test_split(
        all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    
    print(f"  Test set: {len(test_files)} files ({sum(test_labels)} violent, {len(test_labels) - sum(test_labels)} safe)")
    return test_files, test_labels


def run_full_pipeline_evaluation():
    """Run the full CMAG pipeline on every test audio file and compute metrics."""
    output_dir = "performance_metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("FULL PIPELINE EVALUATION — Real Whisper + BERT + CMAG")
    print("=" * 60)
    
    # Step 1: Load test files
    print("\n[1/4] Loading test dataset...")
    test_files, test_labels = get_test_files()
    
    # Step 2: Initialize the full pipeline
    print("\n[2/4] Initializing ViolenceDetectionPipeline...")
    pipeline = ViolenceDetectionPipeline(whisper_model="tiny")
    pipeline.load_weights()
    
    # Step 3: Process each file
    print(f"\n[3/4] Processing {len(test_files)} test files...")
    all_true = []
    all_pred = []
    all_scores = []
    errors = []
    
    start_time = time.time()
    
    for i, (fpath, true_label) in enumerate(zip(test_files, test_labels)):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(test_files) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(test_files)}] {rate:.1f} files/s | ETA: {eta:.0f}s")
        
        try:
            result = pipeline.process_file(fpath)
            pred_label = 1 if result["final_state"] == "VIOLENCE" else 0
            score = result["temporal_score"]
            
            all_true.append(true_label)
            all_pred.append(pred_label)
            all_scores.append(score)
            
        except Exception as e:
            errors.append((fpath, str(e)))
            continue
    
    elapsed_total = time.time() - start_time
    print(f"\n  Completed in {elapsed_total:.1f}s ({len(all_true)} successful, {len(errors)} errors)")
    
    if len(all_true) == 0:
        print("FATAL: No files were processed successfully.")
        return
    
    # Step 4: Compute metrics and generate plots
    print("\n[4/4] Computing metrics and generating plots...")
    
    true_np = np.array(all_true)
    pred_np = np.array(all_pred)
    scores_np = np.array(all_scores)
    
    # Classification report
    report = classification_report(true_np, pred_np, target_names=["Safe", "Violent"], digits=4)
    print("\n" + report)
    
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write("Full Pipeline Evaluation (Whisper STT + BERT + CMAG + Temporal Tracker)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total files: {len(all_true)} | Errors: {len(errors)} | Time: {elapsed_total:.1f}s\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    
    # Save raw results for reproducibility
    raw_results = {
        "total_files": len(all_true),
        "errors": len(errors),
        "elapsed_seconds": round(elapsed_total, 1),
        "true_labels": all_true,
        "pred_labels": all_pred,
        "scores": [float(s) for s in all_scores]
    }
    with open(os.path.join(output_dir, "full_pipeline_results.json"), "w") as f:
        json.dump(raw_results, f, indent=2)
    
    # -- Confusion Matrix --
    cm = confusion_matrix(true_np, pred_np)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Safe", "Violent"], yticklabels=["Safe", "Violent"])
    plt.title('Confusion Matrix — Full CMAG Pipeline')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # -- ROC Curve --
    fpr, tpr, _ = roc_curve(true_np, scores_np)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # -- PR Curve --
    precision_arr, recall_arr, thresholds = precision_recall_curve(true_np, scores_np)
    pr_auc = average_precision_score(true_np, scores_np)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_arr, precision_arr, color='purple', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # -- Calibration Curve --
    try:
        prob_true, prob_pred = calibration_curve(true_np, scores_np, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', color='indigo', label='CMAG Pipeline')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve (Reliability)')
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  [Warning] Calibration curve failed: {e}")
    
    # -- Probability Distribution --
    plt.figure(figsize=(8, 6))
    safe_scores = scores_np[true_np == 0]
    violent_scores = scores_np[true_np == 1]
    plt.hist(safe_scores, bins=30, alpha=0.6, color='#1f77b4', label='Safe', edgecolor='black')
    plt.hist(violent_scores, bins=30, alpha=0.6, color='#d62728', label='Violent', edgecolor='black')
    plt.xlabel('Predicted Violence Probability')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution by Class')
    plt.legend(loc="upper center")
    plt.savefig(os.path.join(output_dir, "probability_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # -- DET Curve --
    fnr = 1 - tpr
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, fnr, color='darkred', lw=2, label='DET Curve')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Positive Rate (FPR) - Log Scale')
    plt.ylabel('False Negative Rate (FNR) - Log Scale')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, "det_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # -- MCC vs Threshold --
    thresholds_mcc = np.linspace(0, 1, 50)
    mcc_scores = [matthews_corrcoef(true_np, (scores_np >= t).astype(int)) for t in thresholds_mcc]
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds_mcc, mcc_scores, color='magenta', lw=2, label='MCC Score')
    plt.xlim([0.0, 1.0])
    plt.ylim([-1.0, 1.05])
    plt.xlabel('Detection Threshold')
    plt.ylabel('MCC Score (-1 to 1)')
    plt.title('Matthews Correlation Coefficient (MCC) vs Threshold')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "mcc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll metrics and plots saved to '{output_dir}/'")
    print(f"Classification report saved to '{output_dir}/classification_report.txt'")
    print(f"Raw results saved to '{output_dir}/full_pipeline_results.json'")


if __name__ == "__main__":
    run_full_pipeline_evaluation()
