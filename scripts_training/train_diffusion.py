"""
Module for training on the difussion models (graphGDP)
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
    # retrieve arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--nb_layer", type=int, default=3, help="Number of GRAN layers")

    # hidden_dim (int)
    parser.add_argument(
        "--hidden_dim", type=int, default=32, help="Hidden dimension of the model"
    )

    # parser about the dataset type
    parser.add_argument(
        "--dataset_type", type=str, default="grid", help="Type of dataset to use"
    )

    # parser about the dataset size
    parser.add_argument(
        "--nb_graphs", type=int, default=2000, help="Number of graphs to generate"
    )

    # batch size
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    # nb epoch training
    parser.add_argument(
        "--nb_epoch", type=int, default=10, help="Number of epoch to train"
    )

    # gpu or cpu (str)
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for training"
    )

    # retrieve the arguments
    args = parser.parse_args()

    if args.dataset_type == "grid":
        # basicly we load a training and a validation dataset
        print("Loading the dataset...")
        training_dataset = DatasetGrid(args.nb_graphs, 10, 10, 1)

        print("Loading the validation dataset...")
        validation_dataset = DatasetGrid(100, 10, 10, 1)

    # we create the dataloader
    training_dataloader = DataLoader(
        training_dataset, batch_size=args.batch_size, shuffle=True
    )

    validation_dataloader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )

    print("Training on graphs")
    model = TrainerGraphGDP(nb_layer=2, hidden_dim=16, nb_max_node=100)

    # we need a custom tensboard logger
    logger = pl.loggers.TensorBoardLogger("logs/", name="diffusion_" + args.dataset_type)

    # adding a checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        mode="min",
    )

    print("Training...")
    # we create the trainer
    trainer = pl.Trainer(
        max_epochs=args.nb_epoch,
        logger=logger,
        accelerator=args.device,
        callbacks=[checkpoint_callback],
    )

    # we train the model
    trainer.fit(model, training_dataloader, validation_dataloader)

    # we save the model
    trainer.save_checkpoint("model.ckpt")
