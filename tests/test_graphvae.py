"""
Test the GraphVAE class.

Basicly the idea is to test the GraphVAE class with a simple graph
"""
import torch

from torch_geometric.data import Data

from deepgraphgen.utils import GraphEncoder, GraphDecoder
from deepgraphgen.graphvae import GraphVAE


def test_encoder():
    graph_encoder = GraphEncoder(nb_layers=2, in_dim=3, hidden_dim=4, out_dim=5)

    # create the graph
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    node_features = torch.randn(3, 3)

    graph = Data(x=node_features, edge_index=edge_index)

    # create 1 batch graph
    graph.batch = torch.tensor([0, 0, 0], dtype=torch.long)

    output_test = graph_encoder(graph)

    assert output_test.shape == (1, 5)


def test_decoder():
    graph_decoder = GraphDecoder(
        nb_layers=2, in_dim=3, hidden_dim=4, out_dim=10 * 10, nb_nodes=10
    )

    input_test = torch.randn(10, 3)

    output_test = graph_decoder(input_test)

    assert output_test.shape == (10, 10, 10)


def test_graphvae():
    """
    Create the full graphvae model and test it
    """

    # init graph encoder
    graph_encoder = GraphEncoder(nb_layers=2, in_dim=3, hidden_dim=4, out_dim=5)

    # init graph decoder
    graph_decoder = GraphDecoder(
        nb_layers=2, in_dim=5, hidden_dim=4, out_dim=10 * 10, nb_nodes=10
    )

    # init graphvae
    graphvae = GraphVAE(graph_encoder, graph_decoder)

    # create the graph
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    node_features = torch.randn(3, 3)

    graph = Data(x=node_features, edge_index=edge_index)

    # create 1 batch graph
    graph.batch = torch.tensor([0, 0, 0], dtype=torch.long)

    # test the forward pass
    latent, proba = graphvae(graph)

    assert latent.shape == (1, 5)

    assert proba.shape == (1, 10, 10)
