import os
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from src.config import SAVED_MODELS_DIR, TEXT_EMBED_DIM, FUSION_HIDDEN_DIM
from src.models.audio_encoder import AudioEncoder
from src.models.cmag_v2 import EnhancedCMAG
from src.models.temporal import TemporalEscalation
from src.data.cached_loader import get_cached_dataloaders
from src.training.train_audio import FocalLoss

def train_temporal(epochs: int = 20, lr: float = 1e-3, patience: int = 5):
    print("=" * 60)
    print("PHASE 5: Training Temporal Escalation (Bi-LSTM)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    audio_encoder = AudioEncoder().to(device)
    audio_ckpt = os.path.join(SAVED_MODELS_DIR, "audio_encoder.pth")
    if os.path.exists(audio_ckpt):
        audio_encoder.load_state_dict(torch.load(audio_ckpt, map_location=device))
    audio_encoder.eval()
    
    cmag = EnhancedCMAG().to(device)
    cmag_ckpt = os.path.join(SAVED_MODELS_DIR, "cmag_v2.pth")
    if os.path.exists(cmag_ckpt):
        cmag.load_state_dict(torch.load(cmag_ckpt, map_location=device))
    cmag.eval()
    
    text_model_path = os.path.join(SAVED_MODELS_DIR, "nlp_violence_expert")
    if os.path.exists(text_model_path):
        from transformers import AutoModel
        text_encoder = AutoModel.from_pretrained(text_model_path).to(device)
    else:
        raise FileNotFoundError("Phase 3 NLP model not found.")
    text_encoder.eval()
    
    for p in audio_encoder.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False
    for p in cmag.parameters():
        p.requires_grad = False
    
    temporal = TemporalEscalation(input_dim=FUSION_HIDDEN_DIM * 2).to(device)
    optimizer = optim.AdamW(temporal.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    criterion = FocalLoss(gamma=2.0)
    
    train_loader, test_loader = get_cached_dataloaders(batch_size=32)
    seq_len = 5
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        temporal.train()
        train_loss = 0.0
        batch_count = 0
        feature_buffer, label_buffer = [], []
        
        for mel_specs, labels in train_loader:
            mel_specs, labels = mel_specs.to(device), labels.to(device)
            with torch.no_grad():
                audio_emb = audio_encoder(mel_specs)
                text_emb = torch.zeros(mel_specs.size(0), TEXT_EMBED_DIM).to(device)
                _, fused = cmag(audio_emb, text_emb, return_features=True)
            
            for i in range(mel_specs.size(0)):
                feature_buffer.append(fused[i])
                label_buffer.append(labels[i])
                if len(feature_buffer) >= seq_len:
                    seq_features = torch.stack(feature_buffer[:seq_len]).unsqueeze(0)
                    seq_labels = torch.stack(label_buffer[:seq_len]).max().unsqueeze(0)
                    
                    optimizer.zero_grad()
                    output = temporal(seq_features)
                    loss = criterion(output, seq_labels.float().unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    batch_count += 1
                    feature_buffer.pop(0)
                    label_buffer.pop(0)
        
        if batch_count > 0:
            train_loss /= batch_count
        
        temporal.eval()
        val_loss, val_count, val_preds, val_targets = 0.0, 0, [], []
        feature_buffer, label_buffer = [], []
        
        with torch.no_grad():
            for mel_specs, labels in test_loader:
                mel_specs, labels = mel_specs.to(device), labels.to(device)
                audio_emb = audio_encoder(mel_specs)
                text_emb = torch.zeros(mel_specs.size(0), TEXT_EMBED_DIM).to(device)
                _, fused = cmag(audio_emb, text_emb, return_features=True)
                
                for i in range(mel_specs.size(0)):
                    feature_buffer.append(fused[i])
                    label_buffer.append(labels[i])
                    if len(feature_buffer) >= seq_len:
                        seq_features = torch.stack(feature_buffer[:seq_len]).unsqueeze(0)
                        seq_labels = torch.stack(label_buffer[:seq_len]).max().unsqueeze(0)
                        
                        output = temporal(seq_features)
                        loss = criterion(output, seq_labels.float().unsqueeze(1))
                        val_loss += loss.item()
                        val_count += 1
                        val_preds.extend((output > 0.5).cpu().numpy().flatten())
                        val_targets.extend(seq_labels.cpu().numpy().flatten())
                        feature_buffer.pop(0)
                        label_buffer.pop(0)
        
        if val_count > 0:
            val_loss /= val_count
            val_acc = accuracy_score(val_targets, val_preds)
            val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        else:
            val_acc = val_f1 = 0.0
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(temporal.state_dict(), os.path.join(SAVED_MODELS_DIR, "temporal_model.pth"))
            print("  → Saved best temporal model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  → Early stopping at epoch {epoch+1}")
                break
        scheduler.step(epoch)
    return temporal

if __name__ == "__main__":
    train_temporal()
