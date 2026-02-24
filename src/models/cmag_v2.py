import torch
import torch.nn as nn
from src.config import AUDIO_EMBED_DIM, TEXT_EMBED_DIM, FUSION_HIDDEN_DIM


class EnhancedCMAG(nn.Module):
    """
    Enhanced Cross-Modal Attention Gating (CMAG-v2).
    
    Key improvements over the original CMAG (v1):
    1. Takes FULL embeddings (128-dim audio from CNN, 768-dim text from DistilBERT)
       instead of (20-dim MFCC average, 1-dim scalar NLP score)
    2. Projects both modalities into a shared hidden space before gating
    3. Bidirectional cross-modal gating with residual connections
    4. Dropout regularization throughout for real-world robustness
    5. Returns both final probability AND intermediate fused features
       (needed by the Temporal Escalation module in Phase 5)
    
    Architecture (following architecture-patterns skill — Clean Architecture):
    ┌─────────────────────────────────────────────────┐
    │  Audio Embed (128)  ──► Audio Proj (128→128)    │
    │  Text Embed (768)   ──► Text Proj (768→128)     │
    │                                                  │
    │  Audio Gate = σ(W_ag · text_proj)                │
    │  Text Gate  = σ(W_tg · audio_proj)               │
    │                                                  │
    │  Gated Audio = audio_proj ⊙ audio_gate + residual│
    │  Gated Text  = text_proj  ⊙ text_gate  + residual│
    │                                                  │
    │  Fused = Concat(gated_audio, gated_text) (256)   │
    │  Output = Fusion MLP → Sigmoid                   │
    └─────────────────────────────────────────────────┘
    
    Input: audio_emb (B, 128), text_emb (B, 768)
    Output: (B, 1) violence probability
    Optional: return_features=True returns (output, fused_features)
    """
    
    def __init__(
        self,
        audio_embed_dim: int = AUDIO_EMBED_DIM,
        text_embed_dim: int = TEXT_EMBED_DIM,
        fusion_hidden_dim: int = FUSION_HIDDEN_DIM,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Project both modalities into a shared hidden space
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_embed_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU()
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU()
        )
        
        # Bidirectional cross-modal attention gates
        # Audio gate: text embedding controls what audio features to emphasize
        self.audio_gate = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.Sigmoid()
        )
        
        # Text gate: audio embedding controls what text features to emphasize
        self.text_gate = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion MLP: processes concatenated gated features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Enhanced CMAG-v2 fusion module.
        
        Args:
            audio_emb: Audio embeddings from AudioEncoder, shape (B, audio_embed_dim)
            text_emb: Text embeddings from TextEncoder, shape (B, text_embed_dim)
            return_features: If True, also return intermediate fused features
                             for the Temporal Escalation module
                             
        Returns:
            output: Violence probability, shape (B, 1)
            fused_features (optional): Concatenated gated embeddings, shape (B, fusion_hidden_dim * 2)
        """
        # Project into shared hidden space
        audio_proj = self.audio_projection(audio_emb)  # (B, fusion_hidden_dim)
        text_proj = self.text_projection(text_emb)      # (B, fusion_hidden_dim)
        
        # Bidirectional cross-modal gating
        # Text controls audio: "which audio patterns matter given what was said?"
        audio_gate_values = self.audio_gate(text_proj)          # (B, fusion_hidden_dim)
        gated_audio = audio_proj * audio_gate_values + audio_proj  # Gating + residual connection
        
        # Audio controls text: "which words matter given what was heard?"
        text_gate_values = self.text_gate(audio_proj)            # (B, fusion_hidden_dim)
        gated_text = text_proj * text_gate_values + text_proj    # Gating + residual connection
        
        # Concatenate gated features
        fused = torch.cat([gated_audio, gated_text], dim=1)  # (B, fusion_hidden_dim * 2)
        
        # Final classification
        output = self.fusion_mlp(fused)  # (B, 1)
        
        if return_features:
            return output, fused
        return output
