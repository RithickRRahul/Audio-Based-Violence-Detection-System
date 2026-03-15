"""
Baseline Evaluation Script: SVM and XGBoost

This script extracts static acoustic features (MFCC, Spectral Roll-off)
and linguistic features (TF-IDF from heuristic filenames) to train and
evaluate the classical ML baselines (SVM and XGBoost).
"""
import os
import sys
import time
import numpy as np
import librosa
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.datasets import load_vsd_dataset, load_cremad_dataset, load_esc50_dataset, load_urbansound_dataset

def get_heuristic_text(filepath):
    """Generate the heuristic fake text used for previous baselines."""
    fname = os.path.basename(filepath).lower()
    if 'noviolence' in fname or 'safe' in fname:
        return 'calm speech'
    elif any(word in fname for word in ['angry', 'fight', 'scream', 'gun', 'violence', 'attack']):
        return 'help angry attack'
    else:
        return 'calm speech'

def extract_features(fpath, label):
    """Extract Audio (MFCC+Rolloff) and Text (heuristic) features."""
    try:
        y, sr = librosa.load(fpath, sr=16000, mono=True, duration=4.0)
        if len(y) == 0:
            y = np.zeros(16000 * 4)
    except Exception:
        y = np.zeros(16000 * 4)
        sr = 16000

    # MFCC (20 dims, mean across time)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Spectral Rolloff (1 dim, mean across time)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff, axis=1)

    audio_features = np.concatenate([mfcc_mean, rolloff_mean])
    text = get_heuristic_text(fpath)
    
    return audio_features, text, label

def load_all_files():
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
        
    # Same split as CMAG test evaluation
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    return train_files, test_files, train_labels, test_labels

def main():
    print("Loading file paths...")
    train_files, test_files, train_labels, test_labels = load_all_files()
    
    print(f"Extracting features for {len(train_files)} training samples...")
    start = time.time()
    train_results = Parallel(n_jobs=-1, batch_size=100)(
        delayed(extract_features)(f, l) for f, l in zip(train_files, train_labels)
    )
    print(f"Extraction took {time.time() - start:.1f}s")
    
    X_train_audio = np.array([res[0] for res in train_results])
    X_train_text_raw = [res[1] for res in train_results]
    y_train = np.array([res[2] for res in train_results])
    
    print(f"Extracting features for {len(test_files)} testing samples...")
    test_results = Parallel(n_jobs=-1, batch_size=100)(
        delayed(extract_features)(f, l) for f, l in zip(test_files, test_labels)
    )
    X_test_audio = np.array([res[0] for res in test_results])
    X_test_text_raw = [res[1] for res in test_results]
    y_test = np.array([res[2] for res in test_results])
    
    # TF-IDF on text
    vectorizer = TfidfVectorizer(max_features=100)
    X_train_text = vectorizer.fit_transform(X_train_text_raw).toarray()
    X_test_text = vectorizer.transform(X_test_text_raw).toarray()
    
    X_train = np.hstack([X_train_audio, X_train_text])
    X_test = np.hstack([X_test_audio, X_test_text])
    
    print(f"Combined Feature Shape: {X_train.shape}")
    
    # ------------------ XGBOOST ------------------
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_p, xgb_r, xgb_f1, _ = precision_recall_fscore_support(y_test, xgb_pred, average='weighted')
    print(f"XGBoost | Acc: {xgb_acc:.4f} | P: {xgb_p:.4f} | R: {xgb_r:.4f} | F1: {xgb_f1:.4f}")
    
    # ------------------ SVM ------------------
    print("Training SVM...")
    # Scale audio features for SVM
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = np.hstack([scaler.fit_transform(X_train_audio), X_train_text])
    X_test_scaled = np.hstack([scaler.transform(X_test_audio), X_test_text])
    
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_p, svm_r, svm_f1, _ = precision_recall_fscore_support(y_test, svm_pred, average='weighted')
    print(f"SVM     | Acc: {svm_acc:.4f} | P: {svm_p:.4f} | R: {svm_r:.4f} | F1: {svm_f1:.4f}")

    print("\nSUMMARY_METRICS")
    print(f"XGBoost,{xgb_acc:.4f},{xgb_p:.4f},{xgb_r:.4f},{xgb_f1:.4f}")
    print(f"SVM,{svm_acc:.4f},{svm_p:.4f},{svm_r:.4f},{svm_f1:.4f}")

if __name__ == "__main__":
    main()
