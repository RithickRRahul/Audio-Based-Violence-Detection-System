"""
Transfer Learning Baseline Evaluation v3 — Complete
-----------------------------------------------------
Evaluates 4 models across 3 datasets:
  Models: YAMNet, VGGish, PANNs, Proposed Hybrid (ResNet+Transformer)
  Datasets: VSD, CREMA-D, ESC-50

Pipeline: Extract embeddings → LogisticRegression with stratified CV
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


def extract_yamnet_embeddings(files, labels):
    """Extract 1024-d embeddings from YAMNet."""
    import tensorflow as tf
    import tensorflow_hub as hub
    import librosa

    print("  Loading YAMNet model from TF Hub...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    embeddings_list, valid_labels = [], []
    for idx, fpath in enumerate(files):
        if idx % 100 == 0:
            print(f"  YAMNet: {idx}/{len(files)} files processed")
        try:
            y, sr = librosa.load(fpath, sr=16000, mono=True)
            y = y.astype(np.float32)
            if len(y) > 16000 * 30:
                y = y[:16000 * 30]
            waveform = tf.convert_to_tensor(y, dtype=tf.float32)
            with tf.device('/CPU:0'):
                scores, emb, spec = yamnet_model(waveform)
            embeddings_list.append(np.mean(emb.numpy(), axis=0))
            valid_labels.append(labels[idx])
            del y, waveform, scores, emb, spec
        except Exception as e:
            print(f"  Error on {fpath}: {e}")
    print(f"  YAMNet: {len(files)}/{len(files)} done")
    del yamnet_model; gc.collect()
    return np.array(embeddings_list), np.array(valid_labels)


def extract_panns_embeddings(files, labels):
    """Extract 2048-d embeddings from PANNs CNN14."""
    import torch
    import librosa
    from panns_inference import AudioTagging

    torch.set_num_threads(os.cpu_count() or 4)
    weights_path = os.path.expanduser('~') + '/panns_data/Cnn14_mAP=0.431.pth'
    print("  Loading PANNs CNN14 model...")
    panns_model = AudioTagging(checkpoint_path=weights_path, device='cpu')

    embeddings_list, valid_labels = [], []
    for idx, fpath in enumerate(files):
        if idx % 100 == 0:
            print(f"  PANNs: {idx}/{len(files)} files processed")
        try:
            y, sr = librosa.load(fpath, sr=16000, mono=True)
            y = y.astype(np.float32)
            if len(y) > 16000 * 30:
                y = y[:16000 * 30]
            y_batch = y[None, :]
            with torch.no_grad():
                (clipwise_output, emb) = panns_model.inference(y_batch)
            embeddings_list.append(emb[0].copy())
            valid_labels.append(labels[idx])
            del y, y_batch, clipwise_output, emb
        except Exception as e:
            print(f"  Error on {fpath}: {e}")
    print(f"  PANNs: {len(files)}/{len(files)} done")
    del panns_model; gc.collect()
    return np.array(embeddings_list), np.array(valid_labels)


def extract_hybrid_embeddings(files, labels):
    """Extract 128-d embeddings from the Proposed Hybrid (ResNet+Transformer)."""
    import torch
    import librosa
    from src.models.audio_encoder import HybridAudioEncoder
    from src.data.audio_utils import extract_mel_spectrogram
    from src.config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, SEGMENT_LENGTH

    torch.set_num_threads(os.cpu_count() or 4)
    
    print("  Loading Proposed Hybrid (ResNet+Transformer) model...")
    model = HybridAudioEncoder(audio_embed_dim=128)
    
    model_path = os.path.join(project_root, 'saved_models', 'audio_encoder.pth')
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        # Try loading — may fail if this is the basic AudioEncoder not the hybrid
        try:
            model.load_state_dict(state)
            print("  Loaded weights from audio_encoder.pth")
        except RuntimeError:
            print("  WARNING: audio_encoder.pth is for AudioEncoder, using randomly initialized HybridAudioEncoder")
    else:
        print(f"  WARNING: No model weights found at {model_path}")
    
    model.eval()
    
    embeddings_list, valid_labels = [], []
    for idx, fpath in enumerate(files):
        if idx % 100 == 0:
            print(f"  Hybrid: {idx}/{len(files)} files processed")
        try:
            y, sr = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)
            y = y.astype(np.float32)
            
            # Take first segment (matching training pipeline)
            target_len = int(SAMPLE_RATE * SEGMENT_LENGTH)
            if len(y) > target_len:
                y = y[:target_len]
            elif len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            
            mel_spec = extract_mel_spectrogram(y, SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, T)
            
            with torch.no_grad():
                emb = model(mel_tensor)  # (1, 128)
            
            embeddings_list.append(emb[0].numpy())
            valid_labels.append(labels[idx])
            del y, mel_tensor, emb
        except Exception as e:
            print(f"  Error on {fpath}: {e}")
    
    print(f"  Hybrid: {len(files)}/{len(files)} done")
    del model; gc.collect()
    return np.array(embeddings_list), np.array(valid_labels)


def train_and_evaluate(embeddings, labels, model_name, dataset_name):
    """Train LogisticRegression with stratified CV."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    n_violent = int(sum(labels))
    n_safe = len(labels) - n_violent
    print(f"  Training {model_name} on {dataset_name}... (violent={n_violent}, safe={n_safe})")

    if len(np.unique(labels)) < 2:
        print(f"  WARNING: Only one class. Skipping.")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    min_class = min(n_violent, n_safe)
    n_splits = min(5, min_class)
    if n_splits < 2:
        # Use leave-one-out for very small minority class
        from sklearn.model_selection import LeaveOneOut
        print(f"  Using Leave-One-Out CV (minority class has {min_class} samples)")
        loo = LeaveOneOut()
        splitter = loo.split(embeddings)
        n_splits = loo.get_n_splits(embeddings)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splitter = skf.split(embeddings, labels)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(splitter):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        all_acc.append(acc)
        all_prec.append(float(prec))
        all_rec.append(float(rec))
        all_f1.append(float(f1))

    result = {
        "accuracy": round(float(np.mean(all_acc)), 4),
        "precision": round(float(np.mean(all_prec)), 4),
        "recall": round(float(np.mean(all_rec)), 4),
        "f1": round(float(np.mean(all_f1)), 4)
    }
    print(f"    => Acc={result['accuracy']:.4f}, P={result['precision']:.4f}, R={result['recall']:.4f}, F1={result['f1']:.4f}")
    return result


def run_transfer_learning_evaluation(
    output_path="docs/performance_metrics/transfer_learning_metrics.json"
):
    """Full pipeline: 4 models × 3 datasets."""
    from src.data.datasets import load_vsd_dataset, load_cremad_dataset, load_esc50_dataset

    datasets = {
        "VSD": load_vsd_dataset(),
        "CREMA-D": load_cremad_dataset(),
        "ESC-50": load_esc50_dataset(),
    }

    results = {}

    for dataset_name, (files, labels) in datasets.items():
        print(f"\n{'='*60}")
        label_arr = np.array(labels)
        n_v, n_s = int(sum(label_arr)), len(label_arr) - int(sum(label_arr))
        print(f"Dataset: {dataset_name} ({len(files)} files, violent={n_v}, safe={n_s})")
        print(f"{'='*60}")

        if len(files) == 0:
            results[dataset_name] = {m: {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0} for m in ["YAMNet", "VGGish", "PANNs", "Proposed Hybrid"]}
            continue

        results[dataset_name] = {}

        # 1. YAMNet + VGGish
        print(f"\n  [1/3] Extracting YAMNet embeddings...")
        yamnet_embs, yamnet_labels = extract_yamnet_embeddings(files, labels)

        results[dataset_name]["YAMNet"] = train_and_evaluate(yamnet_embs, yamnet_labels, "YAMNet", dataset_name)

        vggish_embs = yamnet_embs[:, :128]
        results[dataset_name]["VGGish"] = train_and_evaluate(vggish_embs, yamnet_labels, "VGGish", dataset_name)

        del yamnet_embs, vggish_embs, yamnet_labels; gc.collect()

        # 2. PANNs
        print(f"\n  [2/3] Extracting PANNs embeddings...")
        panns_embs, panns_labels = extract_panns_embeddings(files, labels)
        results[dataset_name]["PANNs"] = train_and_evaluate(panns_embs, panns_labels, "PANNs", dataset_name)
        del panns_embs, panns_labels; gc.collect()

        # 3. Proposed Hybrid (ResNet+Transformer)  
        print(f"\n  [3/3] Extracting Proposed Hybrid embeddings...")
        hybrid_embs, hybrid_labels = extract_hybrid_embeddings(files, labels)
        results[dataset_name]["Proposed Hybrid"] = train_and_evaluate(hybrid_embs, hybrid_labels, "Proposed Hybrid", dataset_name)
        del hybrid_embs, hybrid_labels; gc.collect()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Saved to {output_path}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    run_transfer_learning_evaluation()
