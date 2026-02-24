import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# Audio configurations
SAMPLE_RATE = 16000
SEGMENT_LENGTH = 4.0  # seconds
SAMPLES_PER_SEGMENT = int(SAMPLE_RATE * SEGMENT_LENGTH)

# Mel-spectrogram parameters
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Model architecture dimensions
AUDIO_EMBED_DIM = 128
TEXT_EMBED_DIM = 768  # DistilBERT
FUSION_HIDDEN_DIM = 128

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 30

# Dataset specific paths
VSD_DIR = os.path.join(DATASETS_DIR, "audio_violence")
CREMAD_DIR = os.path.join(DATASETS_DIR, "cremad")
ESC50_DIR = os.path.join(DATASETS_DIR, "esc50")
URBANSOUND_DIR = os.path.join(DATASETS_DIR, "urbansound8k")
JIGSAW_DIR = os.path.join(DATASETS_DIR, "jigsaw_toxic")

# Output paths
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
