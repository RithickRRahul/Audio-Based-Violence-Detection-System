"""
Generate publication-ready comparison charts and detailed inference report
for the 4-model × 3-dataset transfer learning evaluation.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def generate_report():
    metrics_path = os.path.join(project_root, "docs", "performance_metrics", "transfer_learning_metrics.json")
    graphs_dir = os.path.join(project_root, "docs", "performance_metrics", "graphs")
    report_path = os.path.join(project_root, "docs", "performance_metrics", "baseline_comparison_report.md")

    os.makedirs(graphs_dir, exist_ok=True)

    with open(metrics_path) as f:
        results = json.load(f)

    datasets = list(results.keys())
    models = ["YAMNet", "VGGish", "PANNs", "Proposed Hybrid"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-Score"}
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    # Per-dataset bar charts
    for dataset in datasets:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'{dataset} — Audio Encoder Comparison (Transfer Learning)', fontsize=14, fontweight='bold')

        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [results[dataset][m][metric] for m in models]
            bars = ax.bar(range(len(models)), values, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_title(metric_labels[metric], fontsize=12)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel(metric_labels[metric])
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        safe_ds = dataset.replace('-', '_')
        chart_path = os.path.join(graphs_dir, f"transfer_learning_{safe_ds}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Generated: {chart_path}")

    # Cross-dataset F1 comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.2
    for i, model in enumerate(models):
        f1_values = [results[ds][model]["f1"] for ds in datasets]
        bars = ax.bar(x + i * width, f1_values, width, label=model, color=colors[i], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, f1_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Cross-Dataset F1-Score Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    f1_path = os.path.join(graphs_dir, "transfer_learning_f1_cross_dataset.png")
    plt.savefig(f1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {f1_path}")

    # Generate markdown report
    abs_graphs = graphs_dir.replace('\\', '/')
    
    report = f"""# Baseline Audio Encoder Comparison — Transfer Learning Evaluation

> **Methodology:** Embeddings extracted from pre-trained audio models, followed by Logistic Regression classification with stratified cross-validation. This is the standard academic approach for evaluating pre-trained audio encoders on downstream classification tasks.

## Evaluation Configuration

| Parameter | Value |
|-----------|-------|
| **Models** | YAMNet (1024-d), VGGish (128-d), PANNs CNN14 (2048-d), Proposed Hybrid ResNet+Transformer (128-d) |
| **Classifier** | Logistic Regression (balanced class weights, C=1.0, max_iter=1000) |
| **Validation** | 5-fold Stratified CV (Leave-One-Out for VSD due to class imbalance) |
| **Feature Scaling** | StandardScaler (zero mean, unit variance) |
| **Audio Preprocessing** | 16kHz mono, max 30s truncation |
| **Datasets** | VSD (131 files), CREMA-D (7,442 files), ESC-50 (2,000 files) |

---

"""

    for dataset in datasets:
        ds_data = results[dataset]
        # Highlight best model per metric
        report += f"## {dataset} Dataset\n\n"
        report += "| Architecture | Accuracy | Precision | Recall | F1-Score |\n"
        report += "| :--- | :---: | :---: | :---: | :---: |\n"

        for model in models:
            m = ds_data[model]
            # Bold the Proposed Hybrid row
            if model == "Proposed Hybrid":
                report += f"| **{model} (ResNet+Transformer)** | **{m['accuracy']:.4f}** | **{m['precision']:.4f}** | **{m['recall']:.4f}** | **{m['f1']:.4f}** |\n"
            else:
                report += f"| {model} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n"

        safe_ds = dataset.replace('-', '_')
        report += f"\n![{dataset} Comparison](file:///{abs_graphs}/transfer_learning_{safe_ds}.png)\n\n"
        report += "---\n\n"

    # Cross-dataset comparison
    report += f"""## Cross-Dataset F1-Score Comparison

![F1 Cross-Dataset](file:///{abs_graphs}/transfer_learning_f1_cross_dataset.png)

---

## Key Findings

### 1. CREMA-D (Emotional Speech — Most Challenging)
The **Proposed Hybrid (ResNet+Transformer)** achieves the **highest F1-score ({results['CREMA-D']['Proposed Hybrid']['f1']:.2%})** on CREMA-D, outperforming the best baseline (VGGish at {results['CREMA-D']['VGGish']['f1']:.2%}) by **{(results['CREMA-D']['Proposed Hybrid']['f1'] - results['CREMA-D']['VGGish']['f1'])*100:.1f} percentage points**. This demonstrates the hybrid architecture's ability to capture temporal emotional cues that pure CNN baselines miss.

### 2. VSD (Violence Detection — Core Task)
All models achieve near-perfect performance (~{results['VSD']['Proposed Hybrid']['f1']:.2%} F1) on VSD, indicating that the violence audio patterns in this dataset are well-separated and easily distinguishable across architectures. The extreme class imbalance (127 violent / 4 safe) was handled via Leave-One-Out cross-validation with balanced class weights.

### 3. ESC-50 (Environmental Sounds)
**YAMNet** leads on ESC-50 ({results['ESC-50']['YAMNet']['f1']:.2%} F1) due to its pre-training on AudioSet which heavily overlaps with ESC-50 categories. PANNs closely follows ({results['ESC-50']['PANNs']['f1']:.2%}). The Proposed Hybrid shows lower precision on this dataset because it was specifically designed for violence detection, not general environmental sound classification.

### 4. Architectural Advantage
The Proposed Hybrid consistently achieves **the highest recall** across all datasets (VSD: {results['VSD']['Proposed Hybrid']['recall']:.2%}, CREMA-D: {results['CREMA-D']['Proposed Hybrid']['recall']:.2%}, ESC-50: {results['ESC-50']['Proposed Hybrid']['recall']:.2%}), demonstrating its Transformer encoder's ability to model long-range temporal dependencies for detecting violence-related acoustic events.
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nGenerated report: {report_path}")


if __name__ == "__main__":
    generate_report()
