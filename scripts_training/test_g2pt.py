"""
Module to test the model issue for the training
"""

import os
import sys

import torch

import lightning.pytorch as pl

# import dataloader from torch_geometric
from torch.utils.data import DataLoader, Dataset

# import the deepgraphgen modules that are just above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepgraphgen.pl_trainer_g2pt import TrainerG2PT
from deepgraphgen.datageneration import generate_dataset

class DatasetGrid(Dataset):
    """
    Dataset class for grid_graph graphs
    """

    def __init__(self, nb_graphs, nx, ny, edges_to_nodes_ratio=10):
        self.nb_graphs = nb_graphs
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.edges_to_nodes_ratio = edges_to_nodes_ratio

        self.list_graphs = generate_dataset("grid_graph", nb_graphs, nx=nx, ny=ny)


    def __getitem__(self, idx):
        # select the graph
        graph = self.list_graphs[idx]

        # now we need to get the nodes list (simple range from 0 to n)
        nodes = list(range(self.n))

        # now we need to get the edges list
        edges = [(u, v) for u in nodes for v in nodes if graph[u][v]]

        # convert to tensor
        nodes = torch.tensor(nodes)
        edges = torch.tensor(edges)
        
        # now pad the edges to be of size n*edges_to_nodes_ratio
        edges = torch.nn.functional.pad(edges, (0, self.edges_to_nodes_ratio - 1), value=self.n)

        return {
            "nodes": nodes,
            "edges": edges,
        }


if __name__ == "__main__":

    # load the checkpoint
    model = TrainerG2PT

    # we chech the generation of the graph
    #out = model.generate()

    training_dataset = DatasetGrid(100, 10, 10)
    # we create the dataloader
    training_dataloader = DataLoader(
        training_dataset, batch_size=2, shuffle=True
    )

    for batch in training_dataloader:
        print(batch)
        break



