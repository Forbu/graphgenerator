"""
Module to generate dataset / dataloader for the different type of graph
"""

import networkx as nx

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data, DataLoader

from deepgraphgen.datageneration import generate_dataset


def generate_data_graph(graph, nb_nodes, block_size):
    # now we want to select the subgraph with the first nb_nodes nodes
    graph = nx.subgraph(graph, list(range(nb_nodes)))

    # now we want to go from the adjacent matrix to the edge index
    edge_index = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()

    # we create the node features
    node_features = torch.zeros(nb_nodes, 1)

    # now we create the block indexes (the last block_size nodes)
    block_index = torch.tensor(
        list(range(nb_nodes - block_size, nb_nodes)), dtype=torch.long
    )

    # now we create the block edge indexes (all the edges between the block nodes and the other nodes)
    # todo
    block_edges_index = None

    # create the graph
    graph = Data(x=node_features, edge_index=edge_index)

    # create 1 batch graph
    graph.block_index = block_index
    graph.block_edges_index = block_edges_index

    return graph


class DatasetErdos(Dataset):
    """
    Dataset class for Erdos-Renyi graphs
    """

    def __init__(self, nb_graphs, n, p, block_size):
        self.nb_graphs = nb_graphs
        self.n = n
        self.p = p
        self.block_size = block_size

        self.list_graphs = generate_dataset("erdos_renyi_graph", nb_graphs, n=n, p=p)

    def __len__(self):
        return self.nb_graphs * (self.n - self.block_size)

    def __getitem__(self, idx):
        # select the graph
        graph_idx = idx // (self.n - self.block_size)
        nb_nodes = (
            self.n - (idx % (self.n - self.block_size)) + self.block_size
        )  # nb nodes in the current graph
        graph = self.list_graphs[graph_idx]

        return generate_data_graph(graph, nb_nodes, self.block_size)
