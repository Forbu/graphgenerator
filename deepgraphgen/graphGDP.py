"""
Module for the architecture of the diffusion model (graphGDP) (https://arxiv.org/pdf/2212.01842.pdf)

This module is simpler than the GRAN model because there is no auto regression part
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data

from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import global_mean_pool

from deepgraphgen.utils import MLP, init_weights, MPGNNConv


class GraphGDP(nn.Module):
    """
    Torch class for computing the diffusion process
    """

    def __init__(self, nb_layer, hidden_dim, nb_max_node):
        super(GraphGDP, self).__init__()

        self.nb_layer = nb_layer
        self.hidden_dim = hidden_dim
        self.nb_max_node = nb_max_node

        # setup encoder for real node
        self.encoder_edges = MLP(
            in_dim=1,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
        )

        # time encoding
        self.time_encoder = MLP(
            in_dim=1, out_dim=hidden_dim, hidden_dim=hidden_dim, hidden_layers=2
        )

        # setup graph layers (GATv2Conv)
        self.gnn_global = nn.ModuleList()

        # setup graph layers (GATv2Conv)
        self.gnn_filter = nn.ModuleList()

        for _ in range(self.nb_layer):
            self.gnn_global.append(
                GATv2Conv(2 * hidden_dim, hidden_dim, edge_dim=hidden_dim))
            self.gnn_filter.append(
                GATv2Conv(2 * hidden_dim, hidden_dim, edge_dim=hidden_dim))

        # decoding layer for both generated nodes and edges
        self.decoding_layer_edge = MLP(
            in_dim=hidden_dim * 5, out_dim=1, hidden_dim=hidden_dim, hidden_layers=2
        )

        self.apply(init_weights)

    def forward(self, graph_1, graph_2, t_value):
        """
        Forward class for the GraphGDP model

        Args:
            graph_1 (torch_geometric.data.Data): graph 1 (x : node features (nb_nodes, d), edge_index : edge index)
            graph_2 (torch_geometric.data.Data): graph 2
            t_value (float): time value of size (batch_size, 1)

        """

        # get the number of nodes for each graph
        nb_node_graph_1 = graph_1.x.shape[0]
        nb_node_graph_2 = graph_2.x.shape[0]

        edge_attr_full = graph_1.edge_attr[:, 0].unsqueeze(1)
        edge_attr_partial = graph_2.edge_attr.unsqueeze(1)

        # compute the subgraph_idx (batch_idx) for each graph
        subgraph_idx = graph_1.batch

        # create the time encoding
        t_array_nodes = torch.index_select(
            t_value, 0, subgraph_idx).unsqueeze(1)

        # encode the time
        # print(t_array_nodes.shape)
        t_encoding = self.time_encoder(t_array_nodes)

        graph_1.x = torch.concat((t_encoding, t_encoding), dim=1)
        graph_2.x = torch.concat((t_encoding, t_encoding), dim=1)

        # edge encoding
        edge_encoding_graph_1 = self.encoder_edges(edge_attr_full.float())
        edge_encoding_graph_1_init = edge_encoding_graph_1.clone()
        edge_encoding_graph_2 = self.encoder_edges(edge_attr_partial.float())

        # compute the global representation of the graph
        for i in range(self.nb_layer):

            output_graph_1 = (
                self.gnn_global[i](
                    graph_1.x, graph_1.edge_index, edge_encoding_graph_1)
            )

            output_graph_2 = (
                self.gnn_filter[i](
                    graph_2.x, graph_2.edge_index, edge_encoding_graph_2)
            )

            graph_1.x = torch.concat((output_graph_1, output_graph_2), dim=1)
            graph_2.x = torch.concat((output_graph_1, output_graph_2), dim=1)

        # now we want to compute the updated edge score
        # we need to compute the score for each edge of the graph
        # we will use a MLP to compute this score (decoding layer)
        edges_input_graph_1 = torch.cat(
            (graph_1.x[graph_1.edge_index[0]],
             graph_1.x[graph_1.edge_index[1]], edge_encoding_graph_1_init), dim=1
        )


        edges_features = self.decoding_layer_edge(edges_input_graph_1)

        return edges_features
