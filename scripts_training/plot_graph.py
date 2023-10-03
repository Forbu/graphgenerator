"""
Simple module to plot graph in comparaison of the generated data
"""

import os
import sys

import networkx as nx

import matplotlib.pyplot as plt

import lightning.pytorch as pl

# import dataloader from torch_geometric
from torch_geometric.loader import DataLoader

# import the deepgraphgen modules that are just above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepgraphgen.datageneration import (
    generate_dataset
)
from deepgraphgen.pl_trainer import TrainerGRAN


if __name__ == "__main__":

    # we want to generate an erdos_renyi_graph with 100 nodes and p=0.01
    # and a grid graph with 10x10 nodes
    # and a watts_strogatz_graph with 100 nodes and k=2 and p=0.01

    # generate the dataset for erdos
    dataset_erdos = generate_dataset(
        graph_name="erdos_renyi_graph",
        nb_graphs=1,
        n=100,
        p=0.01,
    )

    # plot the graph
    graph = dataset_erdos[0]
    
    # draw 
    nx.draw(graph, with_labels=True)

    # save the graph
    plt.savefig("erdos_renyi_graph.png")

    # clean the plot
    plt.clf()

    # generate the dataset for grid
    dataset_grid = generate_dataset(
        graph_name="grid_graph",
        nb_graphs=1,
        nx=10,
        ny=10,
    )

    # plot the graph
    graph = dataset_grid[0]

    # draw
    nx.draw(graph, with_labels=True)

    # save the graph
    plt.savefig("grid_graph.png")

    # clean the plot
    plt.clf()


    # generate the dataset for watts_strogatz
    dataset_watts_strogatz = generate_dataset(
        graph_name="watts_strogatz_graph",
        nb_graphs=1,
        n=100,
        k=2,
        p=0.01,
    )

    # plot the graph
    graph = dataset_watts_strogatz[0]

    # draw
    nx.draw(graph, with_labels=True)

    # save the graph
    plt.savefig("watts_strogatz_graph.png")

    # clean the plot
    plt.clf()


    