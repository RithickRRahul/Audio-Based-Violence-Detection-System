"""
Evaluate Audio Encoders on the 20% validation (test) set.
Models: YAMNet, VGGish, PANNs, Proposed Hybrid (ResNet+Transformer)
Extracts embeddings for the train split, fits Logistic Regression, and evaluates on the test split.
"""
import os
import sys
import gc
import json
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data.datasets import load_vsd_dataset, load_cremad_dataset, load_esc50_dataset, load_urbansound_dataset
from scripts.evaluate_transfer_learning import extract_yamnet_embeddings, extract_panns_embeddings, extract_hybrid_embeddings

def main():
    print("Loading all datasets to form the global 80/20 split...")
    all_files, all_labels = [], []
    for name, loader in [
        ("VSD", load_vsd_dataset),
        ("CREMA-D", load_cremad_dataset),
        ("ESC-50", load_esc50_dataset),
        ("UrbanSound8K", load_urbansound_dataset)
    ]:
        files, labels = loader()
        all_files.extend(files)
        all_labels.extend(labels)
        
    print(f"Total files: {len(all_files)}")
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    
    print(f"Train split: {len(train_files)} files. Test split: {len(test_files)} files.")
    
    def evaluate_model(model_name, extract_fn, is_vggish=False):
        print(f"\n{'='*40}\nEvaluating {model_name}\n{'='*40}")
        if not is_vggish:
            print("Extracting train embeddings...")
            X_train, _ = extract_fn(train_files, train_labels)
            print("Extracting test embeddings...")
            X_test, _ = extract_fn(test_files, test_labels)
        else:
            # VGGish is first 128 dims of YAMNet. We will pass the YAMNet embs.
            X_train, X_test = extract_fn
            
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=1.0)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{model_name} Results on 20% Test Set:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        return X_train, X_test, acc, prec, rec, f1

    # 1. YAMNet
    yamnet_train, yamnet_test, y_acc, y_p, y_r, y_f1 = evaluate_model("YAMNet", extract_yamnet_embeddings)
    
    # 2. VGGish (subset of YAMNet)
    vggish_train = yamnet_train[:, :128]
    vggish_test = yamnet_test[:, :128]
    _, _, v_acc, v_p, v_r, v_f1 = evaluate_model("VGGish", (vggish_train, vggish_test), is_vggish=True)
    
    del yamnet_train, yamnet_test, vggish_train, vggish_test; gc.collect()
    
    # 3. PANNs
    _, _, p_acc, p_p, p_r, p_f1 = evaluate_model("PANNs (CNN14)", extract_panns_embeddings)
    gc.collect()
    
    # 4. Proposed Hybrid
    _, _, h_acc, h_p, h_r, h_f1 = evaluate_model("Proposed Hybrid", extract_hybrid_embeddings)
    gc.collect()

    print("\n\n" + "-"*50)
    print("FINAL COMBINED 20% TEST SET RESULTS")
    print("-" * 50)
    print("YAMNet:   Acc={:.4f}, P={:.4f}, R={:.4f}, F1={:.4f}".format(y_acc, y_p, y_r, y_f1))
    print("VGGish:   Acc={:.4f}, P={:.4f}, R={:.4f}, F1={:.4f}".format(v_acc, v_p, v_r, v_f1))
    print("PANNs:    Acc={:.4f}, P={:.4f}, R={:.4f}, F1={:.4f}".format(p_acc, p_p, p_r, p_f1))
    print("Hybrid:   Acc={:.4f}, P={:.4f}, R={:.4f}, F1={:.4f}".format(h_acc, h_p, h_r, h_f1))
    print("-" * 50)

if __name__ == "__main__":
    main()
