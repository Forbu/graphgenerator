"""
Module to generate dataset / dataloader for the different type of graph
"""

import networkx as nx

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data
import torch_geometric.transforms as T

from deepgraphgen.datageneration import generate_dataset
from deepgraphgen.diffusion_generation import (
    add_noise_to_graph,
    compute_mean_value_whole_noise,
    generate_beta_value,
)

undirected_transform = T.ToUndirected()

MAX_BETA = 5.0
MIN_BETA = 0.1

def create_full_graph(graph_noisy, gradiant):
    adjacency_matrix_full = torch.ones_like(graph_noisy).to_sparse()

    edge_index_full = adjacency_matrix_full.indices()
    edge_attr_full = graph_noisy[edge_index_full[0], edge_index_full[1]]
    edge_attr_gradiant = gradiant[edge_index_full[0], edge_index_full[1]]

    data_full = Data(
        x=torch.zeros(graph_noisy.shape[0]),
        edge_index=edge_index_full,
        edge_attr=torch.stack(
            [torch.tensor(edge_attr_full), torch.tensor(edge_attr_gradiant)], dim=1
        ),
    )

    return data_full

def create_partial_graph(graph_noisy):
    adjacency_matrix_partial = graph_noisy >= 0
    adjacency_matrix_partial = adjacency_matrix_partial.to_sparse()
    edge_index_partial = adjacency_matrix_partial.indices()
    edge_attr_partial = graph_noisy[edge_index_partial[0], edge_index_partial[1]]

    data_partial = Data(
        x=torch.zeros(graph_noisy.shape[0]),
        edge_index=edge_index_partial,
        edge_attr=edge_attr_partial,
    )

    return data_partial



class DatasetGrid(Dataset):
    """
    Dataset class for grid_graph graphs
    """

    def __init__(self, nb_graphs, nx, ny, nb_timestep=1000):
        self.nb_graphs = nb_graphs
        self.nx = nx
        self.ny = ny
        self.nb_timestep = nb_timestep
        self.n = nx * ny

        self.list_graphs = generate_dataset("grid_graph", nb_graphs, nx=nx, ny=ny)

        self.t_array = torch.linspace(0, 1, nb_timestep)
        self.beta_values = generate_beta_value(MIN_BETA, MAX_BETA, self.t_array)

        self.mean_values, self.variance_values = compute_mean_value_whole_noise(
            self.t_array, self.beta_values
        )

    def __len__(self):
        return self.nb_graphs * self.nb_timestep

    def __getitem__(self, idx):
        # select the graph
        graph_idx = idx // self.nb_timestep
        timestep = idx % self.nb_timestep

        graph = self.list_graphs[graph_idx]

        # convert graph_grid to adjacency matrix (networkx graph to numpy array)
        grid_adjacency_matrix = nx.adjacency_matrix(graph)
        grid_adjacency_matrix = grid_adjacency_matrix.todense()

        # resize the matrix to be between -1 and 1
        grid_adjacency_matrix = 2 * grid_adjacency_matrix - 1

        # add noise to the graph
        graph_noisy, gradiant = add_noise_to_graph(
            grid_adjacency_matrix,
            self.mean_values[timestep],
            self.variance_values[timestep],
        )

        # convert the matrix to a torch tensor
        graph_noisy = torch.tensor(graph_noisy, dtype=torch.float)

        # create the full graph
        data_full = create_full_graph(graph_noisy, gradiant)

        # graph 2 :
        data_partial = create_partial_graph(graph_noisy)

        return {
            "graph_noisy": graph_noisy,
            "gradiant": torch.tensor(gradiant),
            "beta": self.beta_values[timestep],
            "timestep": self.t_array[timestep],
            "data_full": data_full,
            "data_partial": data_partial,
        }
