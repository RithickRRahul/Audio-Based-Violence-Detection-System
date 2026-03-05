import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch_geometric.data import Batch

from src.config import TEXT_EMBED_DIM, SAVED_MODELS_DIR
from src.models.graph_encoder import DependencyGNN
from src.utils.dependency_parser import text_to_dependency_graph


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
    
    def __init__(self, model_name: str = "albert-base-v2", text_embed_dim: int = TEXT_EMBED_DIM):
        super().__init__()
        
        expert_path = os.path.join(SAVED_MODELS_DIR, "nlp_violence_expert")
        gnn_path = os.path.join(expert_path, "gnn_expert.pth")
        classifier_path = os.path.join(expert_path, "classifier_expert.pth")
        
        if model_name == "roberta-base" and os.path.exists(expert_path):
            model_name = expert_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.text_embed_dim = text_embed_dim
        
        # Add Dependency GNN
        self.gnn = DependencyGNN(in_channels=768, hidden_channels=256, out_channels=256)
        if os.path.exists(gnn_path):
            try:
                self.gnn.load_state_dict(torch.load(gnn_path, map_location="cpu"))
            except RuntimeError:
                print("Warning: GNN weight shape mismatch. Ignored saved weights.")
            
        self.fusion_classifier = nn.Sequential(
            nn.Linear(text_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
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
            fused_embeddings, _ = self(texts)
        return fused_embeddings
    
    def get_threat_score(self, text: str) -> float:
        self.eval()
        with torch.no_grad():
            _, scores = self(texts=[text])
        return scores[0].item()
    
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

        # 2. Generate dependency trees and attach token features
        graphs = [text_to_dependency_graph(t) for t in texts_processed]
        for i, graph in enumerate(graphs):
            nodes_needed = graph.num_nodes
            # Exclude [CLS]
            valid_feats = seq_embeddings[i, 1:]
            
            if valid_feats.size(0) >= nodes_needed:
                graph.x = valid_feats[:nodes_needed]
            else:
                padding = torch.zeros((nodes_needed - valid_feats.size(0), valid_feats.size(1)), device=device)
                graph.x = torch.cat([valid_feats, padding], dim=0)
                
        batch_graph = Batch.from_data_list(graphs).to(device)
        graph_embeddings = self.gnn(batch_graph.x, batch_graph.edge_index, batch_graph.batch)
        
        # 3. Fuse embeddings
        fused_embeddings = torch.cat([text_embeddings, graph_embeddings], dim=1)
        
        # 4. Final Classification
        logits = self.fusion_classifier(fused_embeddings)
        scores = torch.sigmoid(logits)
        
        return fused_embeddings, scores
