"""
Module used to generate the data that will be used to train the score based model (diffusion one)

Basicly the idea of https://arxiv.org/pdf/2212.01842.pdf is to use the diffusion process to generate noisy graph from the data
"""
import numpy as np
import torch

from deepgraphgen.datageneration import generated_graph, generate_dataset


def generate_beta_value(beta_min, beta_max, t_array):
    """
    Function used to generate the beta value for the diffusion process

    β(t) = β¯min + t(β¯max − β¯min) for t ∈ [ 0, 1 ]

    """
    return beta_min + t_array * (beta_max - beta_min)


def compute_mean_value_noise(t_array, whole_beta_values, A0, index_t):
    """
    We have : p0t(At|A0) = N (At; A0e - Int 1/2 R t0 β(s)ds, I - Ie - Int R t0 β(s)ds)
    
    """
    # first we compute the integral - Int(0-t) 1/2 β(s)ds
    integral_beta = 0.5 * np.trapz(whole_beta_values[:index_t], t_array[:index_t])

    # then we compute the integral - Int(0-t) β(s)ds
    integral_beta_2 = np.trapz(whole_beta_values[:index_t], t_array[:index_t])

    mean = A0 * np.exp(-integral_beta)
    variance = 1 - np.exp(- integral_beta_2)

    return mean, variance

def batch_graph(batch_size=32, nb_timesteps=1000):
    """
    Function used to generate a batch of graph (with the same number of nodes)
    """
    list_graph = generate_dataset(
        graph_name="erdos_renyi_graph",
        n=10,
        p=0.5,
        nb_graph=batch_size,
        bfs_order_activation=False,
    )

    # now we want to retrieve the adjacency matrix of each graph
    list_adjacency_matrix = [graph.to_dense() for graph in list_graph]

    # now we want to rescale the adjacency matrix to have values between -1 and 1
    list_adjacency_matrix = [2 * matrix - 1 for matrix in list_adjacency_matrix]

    # now in the idea of diffusion process we want to add noise to each graph
    # so basicly for each graph we want to add a random matrix that is dependent on the time
    # step t (graph specific)

    # we generate the different t value for each graph (value between 0 and 1)
    list_t = np.random.uniform(0, 1, batch_size)

    # then we generate all the beta values
