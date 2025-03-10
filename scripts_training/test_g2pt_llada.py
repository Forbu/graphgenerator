"""
Module to test the model issue for the training
"""

import os
import sys
import networkx as nx


import torch

import lightning.pytorch as pl

# import dataloader from torch_geometric
from torch.utils.data import DataLoader, Dataset

# import tensorboardlogger from pl
from lightning.pytorch.loggers import TensorBoardLogger

# wandb logger
# from lightning.pytorch.loggers import WandbLogger

# import the deepgraphgen modules that are just above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepgraphgen.pl_trainer_g2pt_llada import TrainerG2PT
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

        #self.list_graphs = generate_dataset("watts_strogatz_graph", nb_graphs, n=nx * ny, k=2, p=0.01)
        self.list_graphs = generate_dataset("random_labeled_tree", nb_graphs, n=nx * ny)

    def __len__(self):
        return self.nb_graphs

    def __getitem__(self, idx):
        # select the graph
        graph = self.list_graphs[idx]

        #permutation = torch.randperm(self.n)

        # random rebal relabel_nodes(G, mapping)
        #graph = nx.relabel_nodes(graph, dict(zip(graph.nodes, permutation)))

        # now we need to get the nodes list (simple range from 0 to n)
        nodes = list(range(self.n))

        # now we need to get the edges list
        edges_list = list(graph.edges)

        # preprocessing
        edges_list = [(i, j) for i, j in edges_list]
        edges = torch.tensor(edges_list)
        # convert to tensor
        nodes = torch.tensor(nodes)

        # now pad the edges to be of size n*edges_to_nodes_ratio
        edges = torch.cat(
            [
                edges,
                torch.ones(
                    (self.n * self.edges_to_nodes_ratio - edges.shape[0], 2),
                    dtype=torch.long,
                )
                * self.n,
            ]
        )

        return {
            "nodes": nodes,
            "edges": edges,
        }


if __name__ == "__main__":
    # load the checkpoint
    model = TrainerG2PT()

    # we chech the generation of the graph
    # out = model.generate()

    training_dataset = DatasetGrid(10000, 10, 10)
    # we create the dataloader
    training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=False)

    logger = TensorBoardLogger("tb_logs/", name="g2pt_grid")
    #logger = WandbLogger(project="g2pt_grid")

    # setup trainer
    trainer = pl.Trainer(
        max_time={"hours": 3},
        logger=logger,
        accumulate_grad_batches=4,
        #fast_dev_run=True,
    )

    # train the model
    trainer.fit(model, training_dataloader)
    # model.generation_global(batch_size=4, num_nodes=100)