"""
Module to generate dataset / dataloader for the different type of graph
"""

import networkx as nx

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data
import torch_geometric.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from deepgraphgen.datageneration import generate_dataset
from deepgraphgen.diffusion_generation import (
    add_noise_to_graph,
    compute_mean_value_whole_noise,
    generate_beta_value,
)

undirected_transform = T.ToUndirected()

MAX_BETA = 10.0
MIN_BETA = 0.1
NB_RANDOM_WALK = 4
THRESHOLD = 0.2


def create_full_graph(graph_noisy, gradiant, graph_init=None):
    device = graph_noisy.device

    adjacency_matrix_full = torch.ones_like(graph_noisy).to_sparse().to(device)

    if graph_init is None:
        graph_init = torch.zeros_like(graph_noisy).to(device)

    edge_index_full = adjacency_matrix_full.indices().to(device)
    edge_attr_full = graph_noisy[edge_index_full[0], edge_index_full[1]]
    edge_attr_gradiant = gradiant[edge_index_full[0], edge_index_full[1]]
    edge_attr_init = graph_init[edge_index_full[0], edge_index_full[1]]

    data_full = Data(
        x=torch.zeros(graph_noisy.shape[0]).to(device),
        edge_index=edge_index_full,
        edge_attr=torch.stack(
            [
                torch.tensor(edge_attr_full),
                torch.tensor(edge_attr_gradiant),
                torch.tensor(edge_attr_init),
            ],
            dim=1,
        ),
    )

    return data_full


def compute_random_walk_matrix(adjacency_matrix_partial, degree):
    """
    Function used to compute the random walk embedding information
    """
    # we we create N random walk for each node
    # using the Adjacency matrix / degree
    RW = torch.tensor(adjacency_matrix_partial) / degree.unsqueeze(1)
    RW.fill_diagonal_(0)

    list_RW_matrix = []

    value = RW

    for _ in range(NB_RANDOM_WALK):
        # create the random walk matrix
        value = torch.matmul(RW, value)

        list_RW_matrix.append(value)

    # we concat all the matrix (W, W, NB_RANDOM_WALK)
    RW_matrix = torch.stack(list_RW_matrix, dim=2)
    RW_matrix = RW_matrix

    return RW_matrix


def create_partial_graph(graph_noisy):
    adjacency_matrix_partial = graph_noisy >= THRESHOLD

    # force diagonal to be 0
    adjacency_matrix_partial.fill_diagonal_(0)

    adjacency_matrix_partial = adjacency_matrix_partial.to_sparse()
    edge_index_partial = adjacency_matrix_partial.indices()
    edge_attr_partial = graph_noisy[edge_index_partial[0], edge_index_partial[1]]

    # first we compute the degree of each node in the graph
    degree = (graph_noisy >= 0).sum(dim=1)

    # replace 0 by 1
    degree[degree == 0.0] = 1.0

    RW_matrix = compute_random_walk_matrix(graph_noisy >= 0, degree)

    # we retrieve the node features (the diagonal of the matrix)
    nodes_features = torch.diagonal(RW_matrix, dim1=0, dim2=1).T

    # retrieve the edges features
    edges_features = RW_matrix[edge_index_partial[0], edge_index_partial[1]]

    # now we concat the node features witht the degree information
    nodes_features = torch.cat((nodes_features, degree.unsqueeze(1)), dim=1)

    # we concat the edge features with the edge_attr_partial
    edge_attr_partial = torch.cat(
        (edges_features, edge_attr_partial.unsqueeze(1)), dim=1
    )

    data_partial = Data(
        x=nodes_features,
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
        data_full = create_full_graph(graph_noisy, gradiant, grid_adjacency_matrix)

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
