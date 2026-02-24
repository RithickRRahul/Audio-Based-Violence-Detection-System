import torch
import torch.nn as nn
from src.config import FUSION_HIDDEN_DIM


class TemporalAttention(nn.Module):
    """
    Self-attention mechanism over temporal sequence.
    
    Learns WHICH segments in a sequence are most important for the
    final violence prediction. This solves the key problem with
    the old WMA approach: instead of fixed weights, the model
    learns to focus on the segments that actually signal violence.
    
    Input: (batch, seq_len, hidden_dim) — LSTM outputs
    Output: (batch, hidden_dim) — attention-weighted context vector
            (batch, seq_len) — attention weights (for interpretability)
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim) — weighted sum of LSTM outputs
            weights: (batch, seq_len) — attention weights (sum to 1)
        """
        # Compute attention scores
        scores = self.attention(lstm_outputs).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)             # (batch, seq_len)
        
        # Weighted sum
        context = torch.bmm(
            weights.unsqueeze(1),   # (batch, 1, seq_len)
            lstm_outputs            # (batch, seq_len, hidden_dim)
        ).squeeze(1)               # (batch, hidden_dim)
        
        return context, weights


class TemporalEscalation(nn.Module):
    """
    Advanced Temporal Escalation Model with Bi-LSTM + Self-Attention.
    
    This is the most critical component — previous temporal approaches failed
    because they used post-hoc averaging (WMA) instead of learnable temporal
    modeling. This module fixes that with:
    
    1. **Bi-LSTM** (2 layers) — Captures both past→future and future→past
       temporal dependencies in the segment sequence. LayerNorm is applied
       to stabilize training and prevent vanishing gradients.
       
    2. **Self-Attention** — Learns which specific segments in the sequence
       are most informative for the final prediction. A 3-minute audio file
       might have violence only in a 12-second window; attention learns to
       focus on those segments and ignore the rest.
       
    3. **Per-Segment Scoring** — Can output individual violence scores for
       each 4-second segment (needed by the web UI to show timestamp results).
       
    4. **Residual Connection** — The attention context is combined with the
       LSTM's last hidden state via addition, ensuring gradient flow even
       through very long sequences.
    
    Architecture:
    ┌───────────────────────────────────────────────────────────────┐
    │  Input: (B, seq_len, 256) — sequence of CMAG-v2 features    │
    │                                                               │
    │  Layer Norm → Bi-LSTM(256, 64, layers=2, dropout=0.3)        │
    │       ↓ LSTM outputs (B, seq_len, 128)                       │
    │       ├───► Self-Attention → context (B, 128) + weights       │
    │       └───► Per-Segment MLP → segment_scores (B, seq_len, 1) │
    │                                                               │
    │  context + last_hidden (residual) → MLP → Sigmoid → (B, 1)  │
    └───────────────────────────────────────────────────────────────┘
    
    Input: (batch, seq_len, input_dim) — from CMAG-v2 fused features
    Output: (batch, 1) — file-level violence probability
    """
    
    def __init__(
        self,
        input_dim: int = FUSION_HIDDEN_DIM * 2,  # 256 from CMAG concat
        lstm_hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        
        # Input normalization for training stability
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Bi-LSTM: captures temporal context in both directions
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        lstm_output_dim = lstm_hidden * 2  # Bidirectional doubles the output
        
        # Self-attention over LSTM outputs
        self.attention = TemporalAttention(lstm_output_dim)
        
        # Per-segment scoring branch (for the web UI timestamp display)
        self.segment_scorer = nn.Sequential(
            nn.Linear(lstm_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Final classification head with residual
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_per_segment: bool = False,
        return_attention: bool = False,
        raw_cmag_scores: torch.Tensor = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Temporal Escalation model.
        
        Args:
            x: Sequence of CMAG-v2 fused features, shape (B, seq_len, input_dim)
            return_per_segment: If True, also return per-segment violence scores
            return_attention: If True, also return attention weights
            raw_cmag_scores: Optional (B, seq_len) raw fusion scores to preserve spikes
            
        Returns:
            output: File-level violence probability, shape (B, 1)
            per_segment (optional): Per-segment scores, shape (B, seq_len, 1)
            attn_weights (optional): Attention weights, shape (B, seq_len)
        """
        # Normalize input
        x_norm = self.input_norm(x)  # (B, seq_len, input_dim)
        
        # Bi-LSTM
        lstm_out, (h_n, _) = self.lstm(x_norm)  # lstm_out: (B, seq_len, lstm_hidden*2)
        
        # Self-attention aggregation
        context, attn_weights = self.attention(lstm_out)  # (B, lstm_hidden*2)
        
        # Residual: combine attention context with last hidden state
        h_forward = h_n[-2]  
        h_backward = h_n[-1]  
        last_hidden = torch.cat([h_forward, h_backward], dim=1) 
        
        combined = context + last_hidden  # Residual connection
        
        # Final prediction
        output = self.classifier(combined)  # (B, 1)
        
        if return_per_segment:
            # We calculate what the LSTM thinks of the segment
            temporal_scores = self.segment_scorer(lstm_out)  # (B, seq_len, 1)
            
            # CRITICAL FIX: To prevent short violence spikes (e.g. punches in 0-16s) from 
            # being "smoothed away" by the Bi-LSTM's sequence start, we use
            # a MAX gate between the raw CMAG spike and the temporal context.
            if raw_cmag_scores is not None:
                # Shape matching: temporal is (B, seq_len, 1), cmag is (B, seq_len) -> (B, seq_len, 1)
                cmag_expanded = raw_cmag_scores.unsqueeze(-1)
                
                # Bi-LSTM models often start zero-ish for the first few steps of a sequence.
                # If CMAG detected violence, immediately trust it over the LSTM's confusion.
                segment_scores = torch.max(temporal_scores, cmag_expanded)
            else:
                segment_scores = temporal_scores
                
            return output, segment_scores
        
        if return_attention:
            return output, attn_weights
        
        return output
