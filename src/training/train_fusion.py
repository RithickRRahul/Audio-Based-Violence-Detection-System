import os
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

from src.config import SAVED_MODELS_DIR
from src.models.audio_encoder import AudioEncoder
from src.models.cmag_v2 import EnhancedCMAG
from src.data.cached_loader import get_cached_dataloaders
from src.training.train_audio import FocalLoss

def train_fusion(epochs: int = 20, lr: float = 1e-3, patience: int = 5):
    print("=" * 60)
    print("PHASE 4: Training CMAG-v2 Fusion Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    audio_encoder = AudioEncoder().to(device)
    audio_ckpt = os.path.join(SAVED_MODELS_DIR, "audio_encoder.pth")
    if os.path.exists(audio_ckpt):
        audio_encoder.load_state_dict(torch.load(audio_ckpt, map_location=device))
        print("[Fusion] Loaded pre-trained audio encoder")
    audio_encoder.eval()
    for p in audio_encoder.parameters():
        p.requires_grad = False
    
    text_model_path = os.path.join(SAVED_MODELS_DIR, "nlp_violence_expert")
    if os.path.exists(text_model_path):
        print("[Fusion] Loading pre-trained RoBERTa Physical Distress Expert...")
        tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        text_encoder = AutoModel.from_pretrained(text_model_path).to(device)
    else:
        raise FileNotFoundError("Phase 3 NLP model not found. Run train_nlp.py first.")
        
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    
    cmag = EnhancedCMAG().to(device)
    optimizer = optim.AdamW(cmag.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    criterion = FocalLoss(gamma=2.0)
    
    train_loader, test_loader = get_cached_dataloaders(batch_size=32)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        cmag.train()
        train_loss = 0.0
        
        for mel_specs, labels in train_loader:
            mel_specs, labels = mel_specs.to(device), labels.to(device)
            
            with torch.no_grad():
                audio_emb = audio_encoder(mel_specs)
                import random
                fake_texts = []
                for label in labels:
                    if label == 1:
                        fake_texts.append(random.choice(["Help me!", "Stop hitting me!"]) if random.random() < 0.4 else "")
                    else:
                        fake_texts.append(random.choice(["How are you today?", "Nice weather outside"]) if random.random() < 0.4 else "")
                        
                text_inputs = tokenizer(fake_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                text_outputs = text_encoder(**text_inputs)
                text_emb = text_outputs.last_hidden_state[:, 0, :]
            
            optimizer.zero_grad()
            output = cmag(audio_emb, text_emb)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        cmag.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            import random
            for mel_specs, labels in test_loader:
                mel_specs, labels = mel_specs.to(device), labels.to(device)
                audio_emb = audio_encoder(mel_specs)
                
                fake_texts = []
                for label in labels:
                    if label == 1:
                        fake_texts.append(random.choice(["Help me!", "Stop hitting me!"]) if random.random() < 0.4 else "")
                    else:
                        fake_texts.append(random.choice(["How are you today?", "Nice weather outside"]) if random.random() < 0.4 else "")
                
                text_inputs = tokenizer(fake_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                text_outputs = text_encoder(**text_inputs)
                text_emb = text_outputs.last_hidden_state[:, 0, :]

                output = cmag(audio_emb, text_emb)
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
            torch.save(cmag.state_dict(), os.path.join(SAVED_MODELS_DIR, "cmag_v2.pth"))
            print("  → Saved best CMAG model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

if __name__ == "__main__":
    train_fusion()
