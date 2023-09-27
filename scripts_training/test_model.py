"""
Module to test the model issue for the training
"""


import os
import sys

import argparse

import lightning.pytorch as pl

# import dataloader from torch_geometric
from torch_geometric.loader import DataLoader

# import the deepgraphgen modules that are just above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepgraphgen.datasets_torch import DatasetGrid
from deepgraphgen.pl_trainer import TrainerGRAN

if __name__ == "__main__":
    # retrieve arguments (model checkpoint name)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="/workspaces/graphgenerator/scripts_training/checkpoints/model-epoch=09-val_loss=0.00.ckpt",
        help="Checkpoint name",
    )

    # retrieve the argumentsF
    args = parser.parse_args()

    batch_size = 32

    print("Loading the validation dataset...")
    validation_dataset = DatasetGrid(100, 10, 10, 1)

    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    # loading checkpoint
    model = TrainerGRAN()

    print("Loading the model...")
    # load the checkpoint
    model = model.load_from_checkpoint(args.checkpoint_name)

    # now we want to watch the prediction of the model
    model.eval()

    for batch in validation_dataloader:
        break

    # compute the loss
    loss, edges_prob, edge_attr_imaginary = model.compute_loss(batch)

    print("Loss: ", loss)

    print("Edges prob: ", edges_prob.squeeze())
    print("Edges prob max :", edges_prob.max())
    print("Edges prob min :", edges_prob.min())

    print("Edge attr imaginary: ", edge_attr_imaginary.squeeze())
    print("Edge attr imaginary sum: ", edge_attr_imaginary.sum())

    print(batch)

    # graph generation
    print("Generating a graph...")

    graph = model.model.generate()

    print(graph)

    import networkx as nx
    import matplotlib.pyplot as plt

    nx.draw(graph, with_labels=True)

    plt.savefig("graph.png")

    print("End of the test !")
