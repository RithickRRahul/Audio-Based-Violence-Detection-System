import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from src.config import SAVED_MODELS_DIR

# Constants for NLP Training
MODEL_NAME = "roberta-base"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128
CUSTOM_CSV = os.path.join("datasets", "custom_distress", "train.csv")
OUTPUT_DIR = os.path.join(SAVED_MODELS_DIR, "nlp_violence_expert")


class JigsawDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
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


def main():
    print("=" * 60)
    print("PHASE 3: Fine-Tuning RoBERTa NLP Pipeline")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    try:
        texts, labels = load_distress_data()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )

    # 2. Tokenizer & DataLoaders
    print(f"Loading {MODEL_NAME} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = JigsawDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = JigsawDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 3. Model Setup
    print(f"Loading {MODEL_NAME} model...")
    # Using ForSequenceClassification with 1 label for Regression/Binary BCE
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # We use BCEWithLogitsLoss because model outputs unnormalized logits 
    criterion = torch.nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # Train
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        
        # Use simple progress bar due to huge datasets
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # Convert logits to probabilities then to 0/1 predictions
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
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].unsqueeze(1).to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
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
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            
    # Final Report
    print("\nTraining Complete! Final Classification Report (Validation Set):")
    print(classification_report(val_targets, val_preds, target_names=["Safe", "Toxic"]))

if __name__ == "__main__":
    main()
