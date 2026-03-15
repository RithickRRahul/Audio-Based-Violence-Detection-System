import os
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from src.data.datasets import (
    load_vsd_dataset, load_cremad_dataset, load_esc50_dataset, 
    load_urbansound_dataset, AudioViolenceDataset
)
from src.models.audio_encoder import HybridAudioEncoder
from src.models.cmag_v2 import EnhancedCMAG

def evaluate_individual_datasets(output_path: str = "docs/performance_metrics/domain_metrics.json", test_run: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models directly to avoid Whisper dependency
    audio_encoder = HybridAudioEncoder().to(device)
    cmag = EnhancedCMAG().to(device)
    
    audio_encoder.eval()
    cmag.eval()
    
    # Attempt to load weights if not test run
    if not test_run:
        audio_path = "saved_models/audio_encoder.pth"
        cmag_path = "saved_models/cmag_v2.pth"
        if os.path.exists(audio_path):
            audio_encoder.load_state_dict(torch.load(audio_path, map_location=device), strict=False)
        if os.path.exists(cmag_path):
            state_dict = torch.load(cmag_path, map_location=device)
            model_dict = cmag.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            cmag.load_state_dict(model_dict)

    loaders = {
        "VSD": load_vsd_dataset,
        "CREMA-D": load_cremad_dataset,
        "ESC-50": load_esc50_dataset,
        "UrbanSound8K": load_urbansound_dataset
    }
    
    results = {}
    
    for name, loader_func in loaders.items():
        files, labels = loader_func()
        
        if test_run:
            files, labels = files[:10], labels[:10]  # Take tiny subset for testing
            
        dataset = AudioViolenceDataset(files, labels, augment=False)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for mels, batch_labels in dataloader:
                mels = mels.to(device)
                
                # Audio encoding
                audio_emb = audio_encoder(mels)
                
                # True empty text vector for accurate audio-only ablation
                cmag_empty_emb = cmag.text_projection(torch.zeros(1, 768, device=device)) # Using CMAG's projection directly logic
                
                # Use TextEncoder to get the TRUE empty embedding for the batch size
                from src.models.nlp_encoder import TextEncoder
                text_encoder = TextEncoder().to(device)
                text_emb = text_encoder.get_embeddings([""]).to(device)
                text_emb = text_emb.repeat(mels.size(0), 1)
                
                # Fusion score
                outputs = cmag(audio_emb, text_emb, return_features=False)
                
                preds = (outputs > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_labels.numpy())
                
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        # Calculate metrics if we have data
        if len(all_labels) > 0:
            acc = float(accuracy_score(all_labels, all_preds))
            prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
            
            results[name] = {
                "accuracy": acc,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            }
        else:
            results[name] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            
    # Save JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Domain-specific metrics saved to {output_path}")
    
    # Save Markdown Report in same directory
    output_dir = os.path.dirname(output_path)
    md_path = os.path.join(output_dir, "domain_specific_evaluation.md")
    
    md_content = "# Domain-Specific Audio Evaluation Report\n\n"
    md_content += "Performance of the Hybrid Audio Encoder (with NLP disabled) broken down by individual datasets.\n\n"
    
    for name, m in results.items():
        md_content += f"## {name} Dataset\n"
        md_content += f"- **Accuracy:** {m['accuracy']:.4f}\n"
        md_content += f"- **Precision:** {m['precision']:.4f}\n"
        md_content += f"- **Recall:** {m['recall']:.4f}\n"
        md_content += f"- **F1-Score:** {m['f1']:.4f}\n\n"
        
    with open(md_path, "w") as f:
        f.write(md_content)

    print(f"Domain-specific Markdown report saved to {md_path}")

if __name__ == "__main__":
    evaluate_individual_datasets()
