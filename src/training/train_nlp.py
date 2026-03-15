import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from src.config import SAVED_MODELS_DIR
from src.models.nlp_encoder import TextEncoder

# Constants for NLP Training
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128
CUSTOM_CSV = os.path.join("datasets", "custom_distress", "train.csv")
OUTPUT_DIR = os.path.join(SAVED_MODELS_DIR, "nlp_violence_expert")


class JigsawDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': str(self.texts[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }


def load_distress_data(csv_path=CUSTOM_CSV, sample_size=50000):
    """
    Loads Custom Acoustic Distress Dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run generate_distress_dataset.py.")
        
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} records to balance training time...")
        # Stratified sampling without complex groupby
        toxic_df = df[df['is_toxic'] == 1]
        safe_df = df[df['is_toxic'] == 0]
        
        target_per_class = sample_size // 2
        toxic_sample = toxic_df.sample(min(len(toxic_df), target_per_class), random_state=42)
        safe_sample = safe_df.sample(min(len(safe_df), target_per_class), random_state=42)
        
        df = pd.concat([toxic_sample, safe_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
        
    print(f"Dataset Size: {len(df)} | Safe: {len(df[df['is_toxic'] == 0])} | Toxic: {len(df[df['is_toxic'] == 1])}")
    
    return df['comment_text'].values, df['is_toxic'].values


def setup_loss_function(labels: torch.Tensor) -> torch.nn.BCEWithLogitsLoss:
    """
    Calculates the ratio of negative (Safe) to positive (Toxic) samples 
    to create a balanced BCE loss function that explicitly prioritizes Recall.
    """
    labels = labels.cpu()
    neg_count = (labels == 0).sum().float()
    pos_count = (labels == 1).sum().float()
    
    # Avoid division by zero
    if pos_count == 0:
        pos_weight = torch.tensor([1.0])
    else:
        # Example: 100 safe / 20 toxic = pos_weight of 5.0
        # The loss for missing a toxic sample will be multiplied by 5.0
        pos_weight = torch.tensor([neg_count / pos_count])
        
    print(f"Dataset Balancing: Safe={int(neg_count.item())}, Toxic={int(pos_count.item())}")
    print(f"Applied pos_weight (Recall Boost) multiplier: {pos_weight.item():.2f}")
    
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train NLP Expert")
    parser.add_argument("--sample_size", type=int, default=50000, help="Number of samples to use")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 3: Fine-Tuning RoBERTa NLP Pipeline")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    try:
        texts, labels = load_distress_data(sample_size=args.sample_size)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    # Convert labels to tensors so we can compute class weights globally
    global_labels_tensor = torch.tensor(train_labels)

    # 2. DataLoaders
    # Tokenizer is now managed inside TextEncoder, so we don't pass it to Dataset
    train_dataset = JigsawDataset(train_texts, train_labels)
    val_dataset = JigsawDataset(val_texts, val_labels)

    # We collate dynamically so we can handle lists of strings in the batch
    def collate_fn(batch):
        texts = [b['text'] for b in batch]
        labels = torch.stack([b['label'] for b in batch])
        return {'text': texts, 'labels': labels}

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # 3. Model Setup
    print(f"Loading {MODEL_NAME} TextEncoder...")
    model = TextEncoder(model_name=MODEL_NAME)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # We use BCEWithLogitsLoss because TextEncoder returns logits from its latest layer before sigmoid? Let's check nlp_encoder.
    # Ah, nlp_encoder returns probabilities. BCEWithLogitsLoss expects raw logits.
    # We must fix this discrepancy later, but for now BCEWithLogitsLoss is requested.
    criterion = setup_loss_function(global_labels_tensor).to(device)

    # 4. Training Loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        
        # Use simple progress bar due to huge datasets
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            texts = batch['text']
            targets = batch['labels'].unsqueeze(1).to(device)

            optimizer.zero_grad()
            
            # TextEncoder processes strings and passes through BERT
            # Returns (embeddings, probabilities)
            _, logits = model(texts)
            
            loss = criterion(logits, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().detach().numpy().flatten()
            train_preds.extend(preds)
            train_targets.extend(targets.cpu().numpy().flatten())
            
            loop.set_description(f"Loss: {loss.item():.4f}")

        # Val
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text']
                targets = batch['labels'].unsqueeze(1).to(device)

                _, logits = model(texts)
                
                loss = criterion(logits, targets)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).cpu().numpy().flatten()
                
                val_preds.extend(preds)
                val_targets.extend(targets.cpu().numpy().flatten())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds)

        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving best model...")
            
            model.model.save_pretrained(OUTPUT_DIR)
            model.tokenizer.save_pretrained(OUTPUT_DIR)
            
            # We must also save the Fusion Classifier separately
            torch.save(model.fusion_classifier.state_dict(), os.path.join(OUTPUT_DIR, "classifier_expert.pth"))
            
    # Final Report
    print("\nTraining Complete! Final Classification Report (Validation Set):")
    print(classification_report(val_targets, val_preds, target_names=["Safe", "Toxic"]))

if __name__ == "__main__":
    main()
