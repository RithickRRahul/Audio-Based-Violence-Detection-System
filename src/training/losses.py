import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss algorithm for severely imbalanced datasets.
    Mathematically scales down the loss on highly confident 'easy' predictions,
    preventing the model from overwhelmingly guessing the majority class.
    
    alpha: Weighting factor to address class imbalance (typically <0.5 if the target is rare).
           If alpha=0.25, it applies a 0.25 weight to positive classes and 0.75 to negative classes.
    gamma: Precision focusing factor. Gamma > 1 pushes the model to focus exponentially harder 
           on examples it gets wrong, preventing it from resting on easy safe comments.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Calculate standard binary cross-entropy (unreduced)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Calculate probabilities from logits
        probs = torch.sigmoid(logits)
        
        # p_t represents the probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate alpha weight map
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Apply the focal equation: Alpha * (1 - p_t)^Gamma * BCE
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        # Return the mean loss across the batch
        return focal_loss.mean()
