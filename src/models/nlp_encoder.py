import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import TEXT_EMBED_DIM, SAVED_MODELS_DIR


class TextEncoder(nn.Module):
    """
    Text encoder using DistilBERT/RoBERTa for rich semantic embeddings.
    
    Updated to use the trained violence expert model from saved_models
    which contains both the text embeddings and the classification head.
    
    Input: list of strings (transcribed text segments)
    Output: (batch, text_embed_dim=768) dense embeddings
    
    Also provides:
    - get_threat_score(text) -> float (toxicity probability via classification head)
    - get_embeddings(texts) -> tensor (768-dim embeddings for fusion)
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", text_embed_dim: int = TEXT_EMBED_DIM):
        super().__init__()
        
        expert_path = os.path.join(SAVED_MODELS_DIR, "nlp_violence_expert")
        classifier_path = os.path.join(expert_path, "classifier_expert.pth")
        
        if model_name == "bert-base-uncased" and os.path.exists(expert_path):
            model_name = expert_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.text_embed_dim = text_embed_dim

        self.fusion_classifier = nn.Sequential(
            nn.Linear(text_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        if model_name in ["bert-base-uncased", expert_path]:
            if os.path.exists(classifier_path):
                try:
                    self.fusion_classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
                except RuntimeError:
                    print("Warning: Classifier weight shape mismatch. Ignored saved weights.")
    
    def _preprocess(self, texts: list[str]) -> list[str]:
        processed = []
        for text in texts:
            text = text.strip()
            if not text:
                text = "[empty]"
            processed.append(text)
        return processed
    
    def _get_cls_embeddings(self, texts: list[str]) -> torch.Tensor:
        texts = self._preprocess(texts)
        
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        cls_embeddings = outputs.hidden_states[-1][:, 0, :]
        return cls_embeddings
    
    def get_embeddings(self, texts: list[str]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            text_embeddings, _ = self(texts)
        return text_embeddings
    
    def get_threat_score(self, text: str) -> float:
        self.eval()
        with torch.no_grad():
            _, logits = self(texts=[text])
            score = torch.sigmoid(logits)[0].item()
        return score
    
    def forward(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        texts_processed = self._preprocess(texts)
        
        inputs = self.tokenizer(
            texts_processed, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 1. Get semantic text embeddings via Transformer
        outputs = self.model(**inputs, output_hidden_states=True)
        seq_embeddings = outputs.hidden_states[-1]
        text_embeddings = seq_embeddings[:, 0, :]  # CLS token representation

        # 2. Final Classification (returning logits)
        logits = self.fusion_classifier(text_embeddings)
        
        return text_embeddings, logits
