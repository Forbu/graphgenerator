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

from deepgraphgen.datasets_diffusion import (
    DatasetGrid,
)

from deepgraphgen.pl_trainer import TrainerGraphGDP

if __name__ == "__main__":

    model_path = "/workspaces/graphgenerator/scripts_training/checkpoints/model-epoch=01-val_loss=0.07-v2.ckpt"

    # load the checkpoint
    model = TrainerGraphGDP.load_from_checkpoint(model_path)

    model.eval()

    # we chech the generation of the graph
    #out = model.generate()

    training_dataset = DatasetGrid(100, 10, 10)

    # we create the dataloader
    training_dataloader = DataLoader(
        training_dataset, batch_size=1, shuffle=True
    )
