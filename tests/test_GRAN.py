"""
Module to test the GRAN class

The GRAN is an autoregressive model that generates a graph by adding one block of nodes at a time

"""
import pytest

import networkx as nx

import matplotlib.pyplot as plt

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
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    node_features = torch.randn(3, 3)

    # create the block indexes
    block_index = torch.tensor([2], dtype=torch.long)

    # create the block edge indexes

    edge_imaginary_index = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)

    graph = Data(x=node_features, edge_index=edge_index)

    # create 1 batch graph
    graph.batch = torch.tensor([0, 0, 0], dtype=torch.long)

    graph.block_index = block_index

    graph.edge_imaginary_index = edge_imaginary_index

    return graph


def test_gran(input_graph):

    nb_k = 20
    # init the model
    gran = GRAN(
        nb_layer=2,
        in_dim_node=3,
        out_dim_node=3,
        hidden_dim=4,
        nb_max_node=10,
        dim_order_embedding=5,
        nb_k=nb_k
    )

    # forward pass
    nodes_features, edges_prob, global_pooling_values = gran(input_graph)

    assert nodes_features.shape == (1, 3)
    assert edges_prob.shape == (2, nb_k)
    assert global_pooling_values.shape == (1, nb_k)


def test_sampling_graph():
    """
    Function to test the sampling of a graph with the model
    """
    # init the model
    gran = GRAN(
        nb_layer=2,
        in_dim_node=1,
        out_dim_node=3,
        hidden_dim=4,
        nb_max_node=10,
        dim_order_embedding=5,
    )

    # sampling a graph
    graph = gran.generate()

    # plot the networkx graph
    nx.draw(graph, with_labels=True)

    # save the graph (matplotlib)
    plt.savefig("tests/graph.png")
