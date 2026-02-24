import torch
import torch.nn as nn
import torchvision.models as models
from src.config import AUDIO_EMBED_DIM

class AudioEncoder(nn.Module):
    """
    ResNet-18 encoder for mel-spectrograms (Industry Standard).
    Input: (batch, 1, n_mels, time_steps)  — e.g., (B, 1, 128, 126)
    Output: (batch, audio_embed_dim)       — e.g., (B, 128)
    """
    def __init__(self, audio_embed_dim: int = AUDIO_EMBED_DIM):
        super().__init__()
        
        # Load pre-trained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify first conv layer to accept 1 channel (grayscale mel-spec) instead of 3 (RGB)
        # We sum the weights across the RGB channels to maintain pre-trained knowledge
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:] = torch.sum(original_conv1.weight, dim=1, keepdim=True)
            
        # Copy the rest of the ResNet backbone
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Replace the final fully connected layer (resnet.fc) with our custom head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, audio_embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_steps)
        Returns:
            Tensor of shape (batch, audio_embed_dim)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.head(x)
        return x
