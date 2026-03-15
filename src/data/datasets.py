import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from src.data.audio_utils import load_audio, segment_audio, extract_mel_spectrogram
from src.config import (
    SAMPLE_RATE, SEGMENT_LENGTH, N_MELS, N_FFT, HOP_LENGTH,
    VSD_DIR, CREMAD_DIR, ESC50_DIR, URBANSOUND_DIR
)


class AudioViolenceDataset(Dataset):
    """
    Unified audio dataset that loads from any of the 4 audio datasets.
    Each sample returns a mel-spectrogram tensor and a binary label.
    
    Labels: 0 = Safe, 1 = Violent
    """
    
    def __init__(self, file_paths: list[str], labels: list[int], augment: bool = False):
        """
        Args:
            file_paths: List of absolute paths to audio files
            labels: List of binary labels (0=Safe, 1=Violent)
            augment: Whether to apply audio augmentation (training only)
        """
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.augmenter = None
        
        if augment:
            try:
                from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
                self.augmenter = Compose([
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
                    PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
                ])
            except ImportError:
                print("[Warning] audiomentations not available, skipping augmentation")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            y, sr = load_audio(file_path, target_sr=SAMPLE_RATE)
        except Exception as e:
            print(f"[Warning] Failed to load {file_path}: {e}")
            # Return a silent segment
            y = np.zeros(int(SAMPLE_RATE * SEGMENT_LENGTH))
            sr = SAMPLE_RATE
        
        # Take first segment (or pad if too short)
        segments = segment_audio(y, sr, seg_len=SEGMENT_LENGTH)
        y_seg = segments[0]
        
        # Apply augmentation if training
        if self.augment and self.augmenter:
            y_seg = self.augmenter(samples=y_seg, sample_rate=sr)
        
        # Extract mel-spectrogram
        mel_spec = extract_mel_spectrogram(y_seg, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # Convert to tensor: (1, n_mels, time_steps) for CNN input
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
        label_tensor = torch.FloatTensor([label])
        
        return mel_tensor, label_tensor


def load_vsd_dataset(vsd_dir: str = VSD_DIR) -> tuple[list[str], list[int]]:
    """Load the VSD (Violence Sound Detection) dataset."""
    audio_dir = os.path.join(vsd_dir, "audios_VSD")
    files, labels = [], []
    
    if not os.path.exists(audio_dir):
        print(f"[Warning] VSD audio dir not found: {audio_dir}")
        return files, labels
    
    # Walk recursively to handle nested directory structures
    for root, dirs, fnames in os.walk(audio_dir):
        for fname in fnames:
            if not fname.endswith('.wav'):
                continue
            fpath = os.path.join(root, fname)
            fname_lower = fname.lower()
            # Map: noviolence → safe; angry, fight, scream, gun → violent
            # Check 'noviolence' first to prevent 'violence' keyword matching it
            if fname_lower.startswith('noviolence'):
                labels.append(0)
            elif any(kw in fname_lower for kw in ['angry', 'fight', 'scream', 'gun', 'violence', 'attack']):
                labels.append(1)
            else:
                labels.append(0)
            files.append(fpath)
    
    return files, labels


def load_cremad_dataset(cremad_dir: str = CREMAD_DIR) -> tuple[list[str], list[int]]:
    """
    Load CREMA-D emotional speech dataset.
    Labels: ANG (angry) → violent, others → safe
    Filename format: XXXX_XXXXX_ANG_XX.wav
    """
    audio_dir = os.path.join(cremad_dir, "AudioWAV")
    files, labels = [], []
    
    if not os.path.exists(audio_dir):
        print(f"[Warning] CREMA-D dir not found: {audio_dir}")
        return files, labels
    
    for fname in os.listdir(audio_dir):
        if not fname.endswith('.wav'):
            continue
        fpath = os.path.join(audio_dir, fname)
        # CREMA-D filename: ID_SENTENCE_EMOTION_LEVEL.wav
        parts = fname.replace('.wav', '').split('_')
        if len(parts) >= 3:
            emotion = parts[2]
            labels.append(1 if emotion == 'ANG' else 0)
        else:
            labels.append(0)
        files.append(fpath)
    
    return files, labels


def load_esc50_dataset(esc50_dir: str = ESC50_DIR) -> tuple[list[str], list[int]]:
    """
    Load ESC-50 environmental sound classification dataset.
    Violent categories: gunshot, glass_breaking, chainsaw, siren
    """
    csv_path = os.path.join(esc50_dir, "esc50.csv")
    audio_dir = os.path.join(esc50_dir, "audio")
    files, labels = [], []
    
    if not os.path.exists(csv_path):
        print(f"[Warning] ESC-50 CSV not found: {csv_path}")
        return files, labels
    
    violent_categories = {'gunshot', 'glass_breaking', 'chainsaw', 'siren', 'hand_saw'}
    
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        fname = row['filename']
        # Try direct path first, then search in subdirectories
        fpath = os.path.join(audio_dir, fname)
        if not os.path.exists(fpath):
            # Some ESC-50 distributions put files in fold subfolders
            for fold_dir in os.listdir(audio_dir):
                candidate = os.path.join(audio_dir, fold_dir, fname)
                if os.path.exists(candidate):
                    fpath = candidate
                    break
        
        if os.path.exists(fpath):
            category = row.get('category', '')
            labels.append(1 if category in violent_categories else 0)
            files.append(fpath)
    
    return files, labels


def load_urbansound_dataset(us_dir: str = URBANSOUND_DIR) -> tuple[list[str], list[int]]:
    """
    Load UrbanSound8K dataset.
    Violent categories: gun_shot, siren
    """
    base_dir = os.path.join(us_dir, "UrbanSound8K")
    metadata_path = os.path.join(base_dir, "metadata", "UrbanSound8K.csv")
    audio_base = os.path.join(base_dir, "audio")
    files, labels = [], []
    
    if not os.path.exists(metadata_path):
        # Try without subdirectory
        metadata_path2 = os.path.join(us_dir, "metadata", "UrbanSound8K.csv")
        if os.path.exists(metadata_path2):
            metadata_path = metadata_path2
            audio_base = os.path.join(us_dir, "audio")
        else:
            print("[Warning] UrbanSound8K metadata not found")
            return files, labels
    
    violent_classes = {'gun_shot', 'siren'}
    
    df = pd.read_csv(metadata_path)
    for _, row in df.iterrows():
        fold = f"fold{row['fold']}"
        fpath = os.path.join(audio_base, fold, row['slice_file_name'])
        if os.path.exists(fpath):
            class_name = row.get('class', '')
            labels.append(1 if class_name in violent_classes else 0)
            files.append(fpath)
    
    return files, labels


def build_combined_dataset(augment_train: bool = True, test_ratio: float = 0.2):
    """
    Build a combined dataset from all 4 audio sources with balanced sampling.
    
    Returns:
        train_loader, test_loader, class_counts
    """
    print("[Dataset] Loading all datasets...")
    
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
    
    print(f"[Dataset] Total: {len(all_files)} files ({sum(all_labels)} violent, {len(all_labels) - sum(all_labels)} safe)")
    
    # Stratified train/test split
    from sklearn.model_selection import train_test_split
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=test_ratio, stratify=all_labels, random_state=42
    )
    
    # Create datasets
    train_ds = AudioViolenceDataset(train_files, train_labels, augment=augment_train)
    test_ds = AudioViolenceDataset(test_files, test_labels, augment=False)
    
    # Weighted sampler for class imbalance
    train_labels_arr = np.array(train_labels)
    class_counts = np.bincount(train_labels_arr)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels_arr]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    from src.config import BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, test_loader, class_counts
