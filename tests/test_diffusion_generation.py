"""
Test module to test the diffusion generation module.

We will check that the diffusion process is correctly generated (for various graph types)
"""
import pytest

# fixture import
from pytest import fixture

import numpy as np

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
    beta_values = generate_beta_value(0.1, 0.5, t_array)
    assert beta_values.shape == (NB_TIME_STEP,)
    assert beta_values[0] == 0.1
    assert beta_values[-1] == 0.5
    return beta_values


def test_compute_mean_value_whole_noise(t_array, beta_values):
    """
    Test the compute_mean_value_noise function
    """
    mean_values, variance_values = compute_mean_value_whole_noise(t_array, beta_values)
    assert len(mean_values) == NB_TIME_STEP
    assert len(variance_values) == NB_TIME_STEP

