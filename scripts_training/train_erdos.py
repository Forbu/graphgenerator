"""
Module for training on the erdos_renyi_graph dataset
"""

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

from deepgraphgen.datasets_torch import DatasetErdos, DatasetGrid
from deepgraphgen.graphGRAN import GRAN


BATCH_SIZE = 4

if __name__ == "__main__":
    
    # basicly we load a training and a validation dataset
    print("Loading the dataset...")
    training_dataset = DatasetGrid(1000, 10, 10, 2)
    
    print("Loading the validation dataset...")
    validation_dataset = DatasetGrid(100, 10, 10, 2)
    
    # we create the dataloader
    training_dataloader = DataLoader(
        training_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    # we create the model
    class TrainerGRAN(pl.LightningModule):
        """
        Warper class for training
        """
        def __init__(self, nb_layer=3, in_dim_node=1, out_dim_node=1, hidden_dim=32, nb_max_node=100, dim_order_embedding=16):
            super().__init__()
            self.model = GRAN(nb_layer=nb_layer, in_dim_node=in_dim_node, out_dim_node=out_dim_node, hidden_dim=hidden_dim, nb_max_node=nb_max_node, dim_order_embedding=dim_order_embedding)
            
            # init the loss (binary cross entropy)
            self.loss = torch.nn.BCELoss()
            
            # init accuracy metric
            self.train_accuracy = torchmetrics.Accuracy(task="binary")
            
        def forward(self, graphs):
            return self.model(graphs)
        
        def compute_loss(self, graphs):
            """
            Function used to compute the loss
            """
            nodes_features, edges_prob = self.forward(graphs)
            
            # also retrieve the edge_attr_imaginary from the graph to compute the loss
            edge_attr_imaginary = graphs.edge_attr_imaginary
            
            # compute the loss
            loss = self.loss(edges_prob.squeeze(), edge_attr_imaginary.squeeze())
            
            return loss, edges_prob, edge_attr_imaginary
        
        def training_step(self, batch, batch_idx):
            """
            Function used for the training step
            """
            loss, edges_prob, edge_attr_imaginary = self.compute_loss(batch)
            
            # log the loss
            self.log("train_loss", loss)
            
            # log accuracy between the predicted and the real edge
            self.log("train_accuracy", self.train_accuracy(edges_prob.squeeze(), edge_attr_imaginary.squeeze()).cpu().item())
            
            return loss
        
        def validation_step(self, batch, batch_idx):
            """
            Function used for the validation step
            """
            loss, edges_prob, edge_attr_imaginary = self.compute_loss(batch)
            
            # log the loss
            self.log("val_loss", loss)
            
            # log accuracy between the predicted and the real edge
            self.log("val_accuracy", self.train_accuracy(edges_prob.squeeze(), edge_attr_imaginary.squeeze()).cpu().item())
            
            return loss
        
        def configure_optimizers(self):
            """
            Function used to configure the optimizer
            """
            return torch.optim.Adam(self.parameters(), lr=0.001) # trying something different later (Lion)
    
    print("Training on Erdos-Renyi graphs")
    model = TrainerGRAN()
            
    # we need a custom tensboard logger
    logger = pl.loggers.TensorBoardLogger("logs/", name="erdos_renyi")
    
    print("Training...")
    # we create the trainer
    trainer = pl.Trainer(max_epochs=10, logger=logger, accelerator="cpu")
    
    # we train the model
    trainer.fit(model, training_dataloader, validation_dataloader)
    
    # we save the model
    trainer.save_checkpoint("model.ckpt")
            
            
