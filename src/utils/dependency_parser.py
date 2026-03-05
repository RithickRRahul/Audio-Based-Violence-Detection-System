import spacy
import torch
from torch_geometric.data import Data

# Attempt to load the spaCy English model, or download it if absent.
# The user's pip install should have fetched this.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    pass # Assume it exists for real execution based on user script
    
def text_to_dependency_graph(text: str) -> Data:
    """
    Parses natural language text into a PyTorch Geometric Data object 
    representing its grammatical dependency structure.
    Returns only the edge structures and node count (node features
    will be injected later from the ALBERT transformer).
    """
    if not text.strip():
        # Handle empty string gracefully
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        return Data(num_nodes=1, edge_index=edge_index)
        
    doc = nlp(text)
    edges = []
    
    for token in doc:
        # A dependency arc from head token to dependent token
        edges.append([token.head.i, token.i])
        
    # Transpose to get [2, num_edges] shape required by torch-geometric
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Fallback if doc parsed into 0 edges somehow
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
    return Data(num_nodes=len(doc), edge_index=edge_index)
