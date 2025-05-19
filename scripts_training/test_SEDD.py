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

from deepgraphgen.pl_trainer_g2pt_llada_score import TrainerG2PT
from deepgraphgen.datageneration import generate_dataset

from deepgraphgen.datasets_spectre import (
    SpectreGraphDatasetV2,
    DatasetTree,
    DatasetSpectreIterableDataset,
)


import torch._dynamo

torch._dynamo.config.suppress_errors = True

EDGE_TO_NODE_RATIO=3

if __name__ == "__main__":
    # load the checkpoint
    model = TrainerG2PT(
        nb_max_node=100,
        hidden_dim=384,
        nb_layer=6,
        heads=8,
        edges_to_node_ratio=EDGE_TO_NODE_RATIO,
    )

    # compile model
    # model.compile()

    # we chech the generation of the graph
    # out = model.generate()

    # training_dataset = DatasetTree(100000, 10, 10, edges_to_nodes_ratio=3)
    # training_dataset = SpectreGraphDatasetV2(
    #     dataset_name="planar",
    #     download_dir="/app/scripts_training/datasets",
    #     nb_nodes=64,
    #     edges_to_node_ratio=5,
    # )
    training_dataset = DatasetTree(10000, 10, 10, edges_to_nodes_ratio=EDGE_TO_NODE_RATIO)

    iterdataset = DatasetSpectreIterableDataset(dataset=training_dataset)

    # we create the dataloader
    training_dataloader = DataLoader(iterdataset, batch_size=32)

    logger = TensorBoardLogger("tb_logs/", name="g2pt_grid_llada_score")
    # logger = WandbLogger(project="g2pt_grid")

    # model checkpoint
    callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        filename="score_llada_{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_weights_only=True,
        dirpath="models/",
        every_n_train_steps=3000,
        save_last=True,
    )

    # setup trainer
    trainer = pl.Trainer(
        max_time={"hours": 14},
        logger=logger,
        accumulate_grad_batches=4,
        callbacks=[callback],
        # fast_dev_run=True,
        # accelerator="cpu", # debug
        gradient_clip_val=1.0,
        # accelerator="cpu",
    )

    # train the model
    trainer.fit(model, training_dataloader)
    # model.generation_global(batch_size=4, num_nodes=100)
