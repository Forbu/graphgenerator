"""
Module to test the GRAN class

The GRAN is an autoregressive model that generates a graph by adding one block of nodes at a time

"""
import pytest

import torch
from torch_geometric.data import Data

from deepgraphgen.graphGRAN import GRAN

# fixture for inputs of the GRAN model (a graph)
@pytest.fixture
def input_graph():
    """
    Fixture for the input graph
    """
    # create the graph
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    
    node_features = torch.randn(3, 3)
    
    # create the block indexes
    block_index = torch.tensor([2], dtype=torch.long)

    # create the block edge indexes
    block_edge_index = torch.tensor([[1, 2],
                                     [2, 1]], dtype=torch.long)
    
    graph = Data(x=node_features, edge_index=edge_index)
    
    # create 1 batch graph
    graph.batch = torch.tensor([0, 0, 0], dtype=torch.long)
    
    graph.block_index = block_index
    graph.block_edge_index = block_edge_index
    
    return graph

def test_gran(input_graph):
    
    # init the model
    gran = GRAN(nb_layer=2, in_dim_node=3, out_dim_node=3, hidden_dim=4, nb_max_node=10, dim_order_embedding=5)
    
    # forward pass
    nodes_features, edges_prob = gran(input_graph)
    
    assert nodes_features.shape == (1, 3)
    assert edges_prob.shape == (2, 1)
    


