import pytest
import torch

def test_text_encoder_fusion_forward():
    """TDD Step: Test TextEncoder with GNN fusion outputs correct shape."""
    try:
        from src.models.nlp_encoder import TextEncoder
    except ImportError:
        pytest.fail("Cannot import TextEncoder")
        
    model = TextEncoder(model_name="roberta-base")
    model.eval()
    
    texts = [
        "This is a normal sentence.",
        "A violent brutal attack!",
        ""
    ]
    
    # TextEncoder should now internally process graph and text representations
    # and return (embeddings, threat_scores).
    # Since we use pure BERT, new embedding dim should be 768
    
    with torch.no_grad():
        embeddings, scores = model(texts)
        
    assert embeddings.shape == (3, 768), f"Expected (3, 768), got {embeddings.shape}"
    assert scores.shape == (3, 1), f"Expected (3, 1), got {scores.shape}"
