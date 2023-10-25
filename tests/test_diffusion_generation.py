"""
Test module to test the diffusion generation module.

We will check that the diffusion process is correctly generated (for various graph types)
"""
import pytest

# fixture import
from pytest import fixture

import numpy as np

import matplotlib.pyplot as plt

import networkx as nx

from deepgraphgen.diffusion_generation import (
    generate_beta_value,
    compute_mean_value_noise,
    compute_mean_value_whole_noise,
    add_noise_to_graph,
)

from deepgraphgen.datageneration import generated_graph

NB_TIME_STEP = 1000


@fixture
def graph_grid():
    """
    Fixture to generate a graph
    """
    return generated_graph(graph_name="grid_graph", nx=10, ny=10)


@fixture
def graph_watts_strogatz_graph():
    """
    Fixture to generate a graph
    """
    return generated_graph(graph_name="watts_strogatz_graph", n=100, k=4, p=0.1)


@fixture
def t_array():
    """
    Fixture to generate a t_array
    """
    return np.linspace(0, 1, NB_TIME_STEP)


@fixture
def beta_values(t_array):
    """
    Test the generate_beta_value function
    """
    beta_values = generate_beta_value(0.1, 5.0, t_array)
    assert beta_values.shape == (NB_TIME_STEP,)
    assert beta_values[0] == 0.1
    assert beta_values[-1] == 5.0
    return beta_values


def test_compute_mean_value_whole_noise(t_array, beta_values):
    """
    Test the compute_mean_value_whole_noise function
    """
    mean_values, variance_values = compute_mean_value_whole_noise(t_array, beta_values)
    assert len(mean_values) == NB_TIME_STEP
    assert len(variance_values) == NB_TIME_STEP


def test_add_noise_to_graph(graph_grid, t_array, beta_values):
    """
    Test the add_noise_to_graph function
    """
    mean_values, variance_values = compute_mean_value_whole_noise(t_array, beta_values)

    # convert graph_grid to adjacency matrix (networkx graph to numpy array)
    grid_adjacency_matrix = nx.adjacency_matrix(graph_grid)
    grid_adjacency_matrix = grid_adjacency_matrix.todense()

    # resize the matrix to be between -1 and 1
    grid_adjacency_matrix = 2 * grid_adjacency_matrix - 1

    for index_t in range(len(t_array)):
        graph_noisy, gradiant = add_noise_to_graph(
            grid_adjacency_matrix, mean_values[index_t], variance_values[index_t]
        )

        # graph_noisy is a adjacency matrix we want to convert it to a graph (networkx)
        # and then plot it
        graph_noisy = np.array(graph_noisy)

        # we only want to take the upper triangular part of the matrix
        # and make it symmetric
        graph_noisy = np.triu(graph_noisy)

        # we add the transpose of the matrix to make it symmetric
        graph_noisy = graph_noisy + graph_noisy.T

        # the diagonal of the matrix is divided by 2
        # because we have added it twice
        np.fill_diagonal(graph_noisy, np.diagonal(graph_noisy) / 2)

        # we plot the graph
        plt.imshow(graph_noisy)

        #
        plt.colorbar()

        if index_t % 100 == 0:
            print(f"index_t = {index_t}")
            # save in a folder
            plt.savefig(f"tests/figures/figure_{index_t}.png")

        # clear the figure
        plt.clf()
