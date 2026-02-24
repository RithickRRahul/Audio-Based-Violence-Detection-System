import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.config import LEARNING_RATE, EPOCHS, SAVED_MODELS_DIR, AUDIO_EMBED_DIM
from src.models.audio_encoder import AudioEncoder
from src.data.cached_loader import get_cached_dataloaders

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()

def train_audio_encoder(epochs: int = EPOCHS, lr: float = LEARNING_RATE, patience: int = 5):
    print("=" * 60)
    print("PHASE 2: Training Audio Encoder (CNN)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_loader, test_loader = get_cached_dataloaders(batch_size=32)
    print(f"Loaded {len(train_loader)} training batches from cache.")
    
    model = AudioEncoder().to(device)
    classifier = nn.Sequential(
        nn.Linear(AUDIO_EMBED_DIM, 1),
        nn.Sigmoid()
    ).to(device)
    
    optimizer = optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    criterion = FocalLoss(gamma=2.0)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        classifier.train()
        train_loss = 0.0
        
        for mel_specs, labels in train_loader:
            mel_specs, labels = mel_specs.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(mel_specs)
            output = classifier(embeddings)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        classifier.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for mel_specs, labels in test_loader:
                mel_specs, labels = mel_specs.to(device), labels.to(device)
                embeddings = model(mel_specs)
                output = classifier(embeddings)
                loss = criterion(output, labels)
                val_loss += loss.item()
                val_preds.extend((output > 0.5).cpu().numpy().flatten())
                val_targets.extend(labels.cpu().numpy().flatten())
                
        val_loss /= len(test_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVED_MODELS_DIR, "audio_encoder.pth"))
            print("  → Saved best audio model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    model.load_state_dict(torch.load(os.path.join(SAVED_MODELS_DIR, "audio_encoder.pth")))
    print("\n" + classification_report(val_targets, val_preds, target_names=["Safe", "Violent"]))
    return model

if __name__ == "__main__":
    train_audio_encoder()
