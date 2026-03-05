import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class DependencyGNN(nn.Module):
    """
    A Graph Neural Network to process dependency parse trees.
    Takes token embeddings and an adjacency matrix (syntactic edges) 
    and outputs a pooled graph-level embedding.
    """
    def __init__(self, in_channels=768, hidden_channels=256, out_channels=256):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch):
        """
        Args:
            x (Tensor): Node feature matrix `[num_nodes, in_channels]`
            edge_index (LongTensor): Graph connectivity `[2, num_edges]`
            batch (LongTensor): Batch vector `[num_nodes]` defining which graph node is part of
        """
        # First Graph Convolutional layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second Graph Convolutional layer
        x = self.conv2(x, edge_index)
        
        # Global pooling to derive a single embedding per graph in the batch
        x = global_mean_pool(x, batch)
        
        return x
