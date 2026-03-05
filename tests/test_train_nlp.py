import torch
import pytest
from src.training.train_nlp import JigsawDataset
from src.models.nlp_encoder import TextEncoder

def collate_fn(batch):
    texts = [b['text'] for b in batch]
    labels = torch.stack([b['label'] for b in batch])
    return {'text': texts, 'labels': labels}

def test_jigsaw_dataset_collate_and_forward():
    texts = ["I hate you!", "Have a great day", ""]
    labels = [1.0, 0.0, 0.0]
    
    dataset = JigsawDataset(texts, labels)
    assert len(dataset) == 3
    
    batch = [dataset[i] for i in range(3)]
    collated = collate_fn(batch)
    
    assert len(collated['text']) == 3
    assert collated['text'][2] == ""
    assert collated['labels'].shape == (3,)
    
    model = TextEncoder(model_name="roberta-base")
    model.eval()
    
    # Simulate a training forward pass
    with torch.no_grad():
        embeddings, logits = model(collated['text'])
        
    assert embeddings.shape == (3, 1024)
    assert logits.shape == (3, 1)

def test_weighted_loss_function():
    # To boost recall, BCEWithLogitsLoss must be instantiated with pos_weight > 1.0
    from src.training.train_nlp import setup_loss_function
    
    # 100 safe samples, 20 toxic samples -> we want it to penalize toxic misses 5x more
    labels = torch.cat([torch.zeros(100), torch.ones(20)])
    
    criterion = setup_loss_function(labels)
    assert isinstance(criterion, torch.nn.BCEWithLogitsLoss)
    assert criterion.pos_weight is not None
    assert criterion.pos_weight.item() == 5.0

