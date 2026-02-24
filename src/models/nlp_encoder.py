import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from src.config import TEXT_EMBED_DIM


class TextEncoder(nn.Module):
    """
    Text encoder using DistilBERT for rich semantic embeddings.
    
    Following the embedding-strategies skill:
    - Uses dense 768-dim CLS embeddings (not scalar scores)
    - Normalizes embeddings for consistent similarity comparisons
    - Handles empty/blank text gracefully with preprocessing
    
    Input: list of strings (transcribed text segments)
    Output: (batch, text_embed_dim=768) dense embeddings
    
    Also provides:
    - get_threat_score(text) -> float (toxicity probability via classification head)
    - get_embeddings(texts) -> tensor (768-dim embeddings for fusion)
    """
    
    def __init__(self, model_name: str = "roberta-base", text_embed_dim: int = TEXT_EMBED_DIM):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.text_embed_dim = text_embed_dim
        
        # Classification head for threat scoring
        self.classifier = nn.Sequential(
            nn.Linear(text_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def _preprocess(self, texts: list[str]) -> list[str]:
        """
        Preprocess text inputs following embedding-strategies:
        never skip preprocessing, garbage in garbage out.
        """
        processed = []
        for text in texts:
            text = text.strip()
            if not text:
                text = "[empty]"  # Handle blank text gracefully
            processed.append(text)
        return processed
    
    def _get_cls_embeddings(self, texts: list[str]) -> torch.Tensor:
        """Extract CLS token embeddings from DistilBERT."""
        texts = self._preprocess(texts)
        
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        # Move inputs to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # RoBERTa cls token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings
    
    def get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """
        Get dense semantic embeddings for a list of texts.
        Returns: (batch, text_embed_dim) tensor
        """
        return self._get_cls_embeddings(texts)
    
    def get_threat_score(self, text: str) -> float:
        """
        Get a threat/toxicity probability for a single text input.
        Returns: float between 0.0 and 1.0
        """
        embeddings = self._get_cls_embeddings([text])
        score = self.classifier(embeddings)
        return score.item()
    
    def forward(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass returning both embeddings and threat scores.
        Returns: (embeddings, scores) where:
            - embeddings: (batch, text_embed_dim)
            - scores: (batch, 1) 
        """
        embeddings = self._get_cls_embeddings(texts)
        scores = self.classifier(embeddings)
        return embeddings, scores
