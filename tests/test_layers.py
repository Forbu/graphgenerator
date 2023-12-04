"""
In this test module we will test the layers of the model
"""

import torch
from deepgraphgen.utils import TransformerConvEdge

def test_transformerconvedge():
    nb_heads = 15
    nb_features = 3
    nb_nodes = 10
    nb_edges = 10
    nb_edge_features = 5
    nb_output = 4

    layer = TransformerConvEdge(
        in_channels=nb_features,
        out_channels=nb_output,
        heads=nb_heads,
        edge_dim=nb_edge_features,
    )

    # create the graph
    edge_index = torch.randint(0, 6, (2, nb_edges))

    node_features = torch.randn(nb_nodes, nb_features)

    edge_features = torch.randn(nb_edges, nb_edge_features)

    output = layer(node_features, edge_index, edge_features)
    edge_output = layer.out

    assert output.shape == (nb_nodes, nb_output * nb_heads)
    assert edge_output.shape == (nb_edges, nb_heads, nb_output)
