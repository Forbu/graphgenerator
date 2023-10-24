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


def compute_mean_value_noise(t_array, whole_beta_values, index_t):
    """
    We have : p0t(At|A0) = N (At; A0e - Int 1/2 R t0 β(s)ds, I - Ie - Int R t0 β(s)ds)

    """
    # first we compute the integral - Int(0-t) 1/2 β(s)ds
    integral_beta = 0.5 * np.trapz(whole_beta_values[:index_t], t_array[:index_t])

    # then we compute the integral - Int(0-t) β(s)ds
    integral_beta_2 = np.trapz(whole_beta_values[:index_t], t_array[:index_t])

    mean = np.exp(-integral_beta)
    variance = 1 - np.exp(-integral_beta_2)

    return mean, variance


def compute_mean_value_whole_noise(t_array, whole_beta_values):
    """
    Make the computation of the mean value for all the time step
    """
    mean_values = []
    variance_values = []

    for index_t in range(len(t_array)):
        mean, variance = compute_mean_value_noise(t_array, whole_beta_values, index_t)
        mean_values.append(mean)
        variance_values.append(variance)

    return mean_values, variance_values


def add_noise_to_graph(graph, mean_beta, variance):
    """
    Function used to add noise to a graph

    Args:
        graph (torch.tensor): adjacency matrix of the graph
        mean_beta (float): mean value of the noise
        variance (float): variance value of the noise

    Returns:
        graph_noisy (torch.tensor): adjacency matrix of the noisy graph
    """
    # we generate the noise matrix
    mean_beta = graph * mean_beta
    noise_matrix = np.random.normal(mean_beta, np.sqrt(variance), size=(1))

    # we add the noise matrix to the adjacency matrix
    graph_noisy = noise_matrix

    # now we can compute the gradiant of log p0t(At|A0)
    # we have : d/dA0 log p0t(At|A0) = - (At - mean_beta) / variance
    gradiant = -(graph - mean_beta) / variance

    return graph_noisy, gradiant


def batch_graph(
    batch_size=32,
    nb_timesteps=1000,
    list_mean_values=None,
    list_variance_values=None,
):
    """
    Function used to generate a batch of graph (with the same number of nodes)

    Args:
        batch_size (int): number of graph to generate
        nb_timesteps (int): number of time step for the diffusion process
        list_mean_values (list): list of mean values for each graph (diffusion process)
        list_variance_values (list): list of variance values for each graph (diffusion process)

    Returns:
        list_graph (list): list of graph generated (noise added)
        gradiant (list): list of gradiant for each graph (diffusion process)
        array_t (np.array): array of time step for the diffusion process

    """
    list_graph = generate_dataset(
        graph_name="erdos_renyi_graph",
        n=100,
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

    # we choose a random index of t for each graph
    list_index_t = np.random.randint(0, nb_timesteps, size=batch_size)

    # now we select the mean and variance values corresponding to the index t
    list_mean_values = [list_mean_values[index] for index in list_index_t]
    list_variance_values = [list_variance_values[index] for index in list_index_t]

    # now we want to generate the noise matrix for each graph
    list_graph_noisy = []
    gradiant = []

    for index_graph in range(batch_size):
        graph_noisy, grad = add_noise_to_graph(
            list_adjacency_matrix[index_graph],
            list_mean_values[index_graph],
            list_variance_values[index_graph],
        )
        list_graph_noisy.append(graph_noisy)
        gradiant.append(grad)

    t_array = [i / nb_timesteps for i in range(nb_timesteps)]

    return list_graph_noisy, gradiant, t_array
