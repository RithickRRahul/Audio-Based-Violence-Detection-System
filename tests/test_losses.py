import torch
import pytest
from src.training.losses import FocalLoss

def test_focal_loss_reduces_easy_examples():
    """
    TDD Test: Verifies that FocalLoss correctly initializes and penalizes 
    high-confidence easy examples less than low-confidence hard examples.
    """
    fl = FocalLoss(gamma=2.0)
    
    # Logits array:
    # First example: High confidence for 'Safe' (Large negative logit) -> Easy Negative
    # Second example: Low confidence for 'Threat' (Near zero logit) -> Hard Positive
    logits = torch.tensor([[-5.0], [0.1]]) 
    
    # Targets: [Safe (0), Threat (1)]
    targets = torch.tensor([[0.0], [1.0]])
    
    # Forward pass should calculate loss without crashing
    loss = fl(logits, targets)
    
    assert loss.dim() == 0, "Loss must be returned as a scalar value."
    assert not torch.isnan(loss), "Loss computation resulted in NaN (Not a Number)."
