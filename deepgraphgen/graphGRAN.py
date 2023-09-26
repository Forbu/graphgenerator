"""
Based on GRAN paper : https://openreview.net/pdf?id=SJxYOVSgUB

Basicly the idea of the paper is to contruct the graph iteratively with a GNN.

We have two graphs : 
- the real graphs with real informations around the network and all (nodes and edges)
- the imaginary graphs (a block of new nodes with imaginary edges and an imaginary edges connecting the block nodes with the other graphs)

"""

import torch
from torch import nn
import torch.functional as F

from torch_geometric.nn import GATv2Conv

from deepgraphgen.utils import MLP


class GRAN(nn.Module):
    """
    This is a class that will help us autoregressively generate a knowledge graph
    """

    def __init__(
        self,
        nb_layer,
        in_dim_node,
        out_dim_node,
        hidden_dim,
        nb_max_node,
        dim_order_embedding,
    ):
        super().__init__()

        self.nb_layer = nb_layer
        self.in_dim_node = in_dim_node
        self.out_dim_node = out_dim_node
        self.hidden_dim = hidden_dim
        self.nb_max_node = nb_max_node
        self.dim_order_embedding = dim_order_embedding

        # setup encoder for real node
        self.encoder = MLP(
            in_dim=in_dim_node + dim_order_embedding,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
        )

        # setup embedding for fake node
        self.fake_node = nn.Parameter(torch.randn(1, hidden_dim))

        # time embedding / order embedding with xavier_uniform
        self.node_embedding = nn.Parameter(
            torch.randn(nb_max_node, dim_order_embedding)
        )
        nn.init.xavier_uniform_(self.node_embedding)

        # setup graph layers (GATv2Conv)
        self.gnn = nn.ModuleList()

        for i in range(self.nb_layer):
            if i == 0:
                self.gnn.append(GATv2Conv(hidden_dim, hidden_dim))
            elif i == self.nb_layer - 1:
                self.gnn.append(GATv2Conv(hidden_dim, hidden_dim))
            else:
                self.gnn.append(GATv2Conv(hidden_dim, hidden_dim))

        # decoding layer for both generated nodes and edges
        self.decoding_layer_edge = MLP(
            in_dim=hidden_dim * 2, out_dim=1, hidden_dim=hidden_dim, hidden_layers=2
        )
        self.decoding_layer_node = MLP(
            in_dim=hidden_dim,
            out_dim=out_dim_node,
            hidden_dim=hidden_dim,
            hidden_layers=2,
        )

    def forward(
        self,
        graph,
    ):
        """
        Forward pass, with the new nodes block indexes

        Args:
            graph (torch_geometric.data.Data): the graph
                with x (torch.Tensor): the features of the nodes
                and edge_index (torch.Tensor): the edges of the graph
                block_index (torch.Tensor): the indexes of the block nodes

                edge_imaginary_index (torch.Tensor): the indexes of the block edges


        Returns:
            nodes_features (torch.Tensor): the features of the nodes in the block
            edges_prob (torch.Tensor): the probability distribution over the edges of the block

        """
        # retrieve the block indexes
        block_index, edge_imaginary_index = (
            graph.block_index,
            graph.edge_imaginary_index,
        )

        # first node encoding
        nodes = graph.x

        # now we can concat the nodes features with the time embedding
        # for each graph we compute the number of nodes
        nodes_embedding_list = []

        # we loop over all the graphs
        for _ in range(graph.batch.max().item() + 1):
            # we compute the number of nodes in the graph
            nb_nodes = (graph.batch == _).sum().item()

            # we create the node embedding
            nodes_embedding = self.node_embedding[-nb_nodes:]

            # we add the node embedding to the list
            nodes_embedding_list.append(nodes_embedding)

        # concatenate the list
        nodes_embedding = torch.cat(nodes_embedding_list, dim=0)

        # now we can concatenate the nodes features with the time embedding
        nodes = torch.cat([nodes, nodes_embedding], dim=1)
        nodes_features = self.encoder(nodes)

        print("nodes shape", nodes_features.shape)

        # replace the nodes_features in block_index by the fake node
        # nodes_features[block_index] = self.fake_node # no need as there is already information in the time embedding

        # now we can start the graph generation
        for i in range(self.nb_layer):
            nodes_features = self.gnn[i](nodes_features, graph.edge_index)

        # now we can decode the edges
        input_edges = torch.cat(
            [
                nodes_features[edge_imaginary_index[0]],
                nodes_features[edge_imaginary_index[1]],
            ],
            dim=1,
        )

        edges_prob = torch.sigmoid(
            self.decoding_layer_edge(input_edges)
        )  # probabilities

        # now we can decode the nodes
        nodes_features = self.decoding_layer_node(nodes_features[block_index])

        return nodes_features, edges_prob

    def generate(
        self,
    ):
        """
        Generate a graph from (optional) seed node
        """
        pass
