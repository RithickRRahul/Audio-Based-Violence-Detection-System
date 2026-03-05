from src.utils.dependency_parser import text_to_dependency_graph
import torch

def test_graph_has_no_static_features():
    # The new utility should only return edge_index and token count, 
    # not static x features, because ALBERT will provide them later.
    data = text_to_dependency_graph("The quick brown fox.")
    assert not hasattr(data, 'x') or data.x is None
    assert hasattr(data, 'num_nodes')
