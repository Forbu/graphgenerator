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

    β(t) = β¯min + t(β¯max - β¯min) for t ∈ [ 0, 1 ]

    """
    return beta_min + t_array * (beta_max - beta_min)


def compute_mean_value_noise(t_array, whole_beta_values, index_t):
    """
    We have : p0t(At|A0) = N (At; A0e - Int 1/2 R t0 β(s)ds, I - Ie - Int R t0 β(s)ds)

    """
    # first we compute the integral - Int(0-t) 1/2 β(s)ds
    integral_beta = np.trapz(whole_beta_values[:index_t], t_array[:index_t])

    mean = np.exp(- 0.5 * integral_beta)
    variance = 1 - np.exp(- integral_beta)

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

def transform_to_symetric(matrix):
    """
    Function used to transform a matrix to a symetric one

    Args:
        matrix (np.tensor): matrix to transform

    Returns:
        matrix_symetric (np.tensor): symetric matrix
    """
    matrix_symetric = np.triu(matrix, k=0) + np.triu(matrix, k=1).T

    return matrix_symetric

def add_noise_to_graph(graph, mean_beta, variance):
    """
    Function used to add noise to a graph

    Args:
        graph (torch.tensor): adjacency matrix of the graph
        mean_beta (float): mean value of the noise
        variance (float): variance value of the noise

    Returns:
        graph_noisy (torch.tensor): adjacency matrix of the noisy graph
        gradiant (torch.tensor): gradiant of the log p0t(At|A0) for the noisy graph
    """
    # we generate the noise matrix
    mean_beta = graph * mean_beta

    noise_matrix = np.random.normal(mean_beta, np.sqrt(variance), size=mean_beta.shape)
    noise_matrix = transform_to_symetric(noise_matrix)

    # now we can compute the gradiant of log p0t(At|A0)
    # we have : d/dA0 log p0t(At|A0) = - (At - mean_beta) / variance
    if variance == 0:
        gradiant = np.zeros(graph.shape)
    else:
        gradiant = -(noise_matrix - mean_beta) / variance

    return noise_matrix, gradiant
