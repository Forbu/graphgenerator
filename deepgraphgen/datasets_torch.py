"""
Module to generate dataset / dataloader for the different type of graph
"""

import networkx as nx

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data, DataLoader

from deepgraphgen.datageneration import generate_dataset


def generate_data_graph(graph, nb_nodes, block_size):

    # now we create the block indexes (the last block_size nodes)
    block_index = torch.tensor(
        list(range(nb_nodes - block_size, nb_nodes)), dtype=torch.long
    )

    # now we want to select the subgraph with the first nb_nodes nodes
    graph = nx.subgraph(graph, list(range(nb_nodes)))
    
    # convert to list of edges
    edges = list(graph.edges)
    
    if len(edges) == 0:
        return None

    # now we want to go from the adjacent matrix to the edge index
    edge_index_real = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index_real_reverse = torch.tensor(edges[::-1], dtype=torch.long).t().contiguous()
    
    # we want to add the reverse edges
    edge_index_real = torch.cat([edge_index_real, edge_index_real_reverse], dim=1)

    # we want to add all the imaginary edges between the block nodes
    edge_imaginary_index, edge_attr_imaginary = create_imaginary_edges_index(
        nb_nodes, block_size, edge_index_real
    )

    # the full index is the concatenation of the edge index and the imaginary edge index
    edge_index = torch.cat([edge_index_real, edge_imaginary_index], dim=1)

    # drop duplicate edges
    edge_index, _ = torch.unique(edge_index, sorted=True, return_inverse=True, dim=1)

    # we create the node features
    node_features = torch.zeros(nb_nodes, 1)

    # create the graph
    graph = Data(x=node_features, edge_index=edge_index)

    # create 1 batch graph
    graph.block_index = block_index

    graph.edge_imaginary_index = edge_imaginary_index
    graph.edge_attr_imaginary = edge_attr_imaginary


    return graph


def create_imaginary_edges_index(nb_nodes, block_size, edge_index_real):
    """
    Function to create the imaginary edges index
    """
    # we want to add all the imaginary edges between the block nodes
    edge_imaginary_index_list = []
    edge_attr_imaginary_list = []  # todo compute the edge attributes

    for i in range(block_size):
        # create a torch origin tensor (all the nodes)
        torch_origin = torch.arange(0, nb_nodes, dtype=torch.long)

        # torch destination are the block index nodes
        torch_destination = torch.tensor(
            [nb_nodes - block_size + i] * nb_nodes, dtype=torch.long
        )

        # create the edge index
        edge_imaginary_index = torch.stack([torch_origin, torch_destination], dim=0)

        # filter the edge index on the block node
        edge_index_real_block = edge_index_real[
            :, (edge_index_real[0] == nb_nodes - block_size + i)
        ]
        destination_list = edge_index_real_block[1].tolist()

        # init the attribute edge
        edge_attr_imaginary = torch.zeros(nb_nodes, dtype=torch.float)

        # now we just have to check if the edge is in the real graph
        for j in destination_list:
            edge_attr_imaginary[j] = 1

        # add to the list
        edge_imaginary_index_list.append(edge_imaginary_index)
        edge_attr_imaginary_list.append(edge_attr_imaginary)

    # concatenate the list
    edge_imaginary_index = torch.cat(edge_imaginary_index_list, dim=1)

    # concatenate the list
    edge_attr_imaginary = torch.cat(edge_attr_imaginary_list, dim=0)
    
    return edge_imaginary_index, edge_attr_imaginary

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
            self.n - (idx % (self.n - self.block_size))
        )  # nb nodes in the current graph
        graph = self.list_graphs[graph_idx]
        
        resulting_graph = generate_data_graph(graph, nb_nodes, self.block_size)

        if resulting_graph is None:
            return self.__getitem__((idx + 1) % self.__len__())
        else:
            return resulting_graph

class DatasetGrid(Dataset):
    """
    Dataset class for grid_graph graphs
    """
    def __init__(self, nb_graphs, nx, ny, block_size):
        self.nb_graphs = nb_graphs
        self.nx = nx
        self.ny = ny
        self.block_size = block_size
        self.n = nx * ny

        self.list_graphs = generate_dataset("grid_graph", nb_graphs, nx=nx, ny=ny)

    def __len__(self):
        return self.nb_graphs * (self.n - self.block_size)

    def __getitem__(self, idx):
        # select the graph
        graph_idx = idx // (self.n - self.block_size)
        nb_nodes = (
            self.n - (idx % (self.n - self.block_size))
        )  # nb nodes in the current graph
        graph = self.list_graphs[graph_idx]
        
        resulting_graph = generate_data_graph(graph, nb_nodes, self.block_size)

        if resulting_graph is None:
            return self.__getitem__((idx + 1) % self.__len__())
        else:
            return resulting_graph
