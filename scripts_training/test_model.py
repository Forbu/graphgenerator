"""
Module to test the model issue for the training
"""


from deepgraphgen.graphGRAN import GRAN
from deepgraphgen.datasets_torch import DatasetErdos, DatasetGrid
import os
import sys

import argparse

import torch
import lightning.pytorch as pl

# import dataloader from torch_geometric
from torch_geometric.loader import DataLoader

import torchmetrics

# import the deepgraphgen modules that are just above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    