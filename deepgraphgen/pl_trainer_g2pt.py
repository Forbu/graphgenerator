"""
Helper class to train the model (with pytorch lightning)
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import torch
import lightning.pytorch as pl

import matplotlib.pyplot as plt

# we create the model
from x_transformers import Encoder

class TrainerG2PT(pl.LightningModule):
    """
    Warper class for training
    """

    def __init__(
        self,
        nb_layer=6,
        in_dim_node=1,
        out_dim_node=1,
        hidden_dim=386,
        nb_max_node=100,
        edges_to_node_ratio=10
    ):
        super().__init__()

        self.nb_layer = nb_layer
        self.in_dim_node = in_dim_node
        self.out_dim_node = out_dim_node
        self.hidden_dim = hidden_dim
        self.nb_max_node = nb_max_node
        self.edges_to_node_ratio = edges_to_node_ratio

        # transformer core
        self.model_core = Encoder(
            dim = hidden_dim,
            depth = 6,
            heads = 8
        )

        # embedding for the ndoes
        self.nodes_embedding = torch.nn.Embedding(nb_max_node, hidden_dim)

        # linear layer for the edgesinput
        self.edges_embedding = torch.nn.Sequential(
            torch.nn.Linear(2*nb_max_node, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        # linear layer for the time
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        # last layers for nodes edges 0 and 1
        self.nodes_logit = torch.nn.Linear(hidden_dim, nb_max_node)
        self.edges_logit_0 = torch.nn.Linear(hidden_dim, nb_max_node)
        self.edges_logit_1 = torch.nn.Linear(hidden_dim, nb_max_node)


    def forward(self, batch):
        # apply the embedding
        nodes_embedding = self.nodes_embedding(batch["nodes"])
        edges_embedding = self.edges_embedding(batch["edges"])
        time_embedding = self.time_embedding(batch["time"])

        global_embedding = torch.cat([nodes_embedding, edges_embedding], dim=-1)

        # apply film layer with time embedding
        global_embedding = global_embedding * (1. + time_embedding)

        # apply the transformer core
        output_global = self.model_core(global_embedding)

        # angain we apply film layer with time embedding
        output_global = output_global * (1. + time_embedding)

        nodes_output = output_global[:, :self.nb_max_node]
        edges_output = output_global[:, self.nb_max_node:]

        # apply the embedding
        nodes_logit = self.nodes_logit(nodes_output)
        edges_logit_0 = self.edges_logit_0(edges_output)
        edges_logit_1 = self.edges_logit_1(edges_output)

        return nodes_logit, edges_logit_0, edges_logit_1


    def training_step(self, batch, batch_idx):
        """
        Function used for the training step
        nodes_element (torch.Tensor): the elements of the nodes (batch_size, num_nodes)
        edges_element (torch.Tensor): the elements of the edges (batch_size, num_nodes*self.edges_to_node_ratio, 2)
        """
        nodes_element = batch["nodes"]
        edges_element = batch["edges"]

        num_nodes = nodes_element.shape[1]
        batch_size = nodes_element.shape[0]

        nodes_noise, edges_noise = self.noise_generation(batch_size, num_nodes)

        # convert nodes element and edges element to one hot encoding
        nodes_element = torch.nn.functional.one_hot(nodes_element, num_classes=num_nodes)
        edges_element = torch.nn.functional.one_hot(edges_element, num_classes=num_nodes + 1) # shape (batch_size, num_nodes*self.edges_to_node_ratio, 2, num_nodes + 1)

        # flatten the edges element
        edges_element = edges_element.reshape(batch_size, num_nodes*self.edges_to_node_ratio, 2*num_nodes)

        # generate the random noise 
        t = torch.rand(batch_size, 1, 1)

        # generate the interpolation
        nodes_interpolation = t * nodes_element + (1 - t) * nodes_noise
        edges_interpolation = t * edges_element + (1 - t) * edges_noise

        batch = {
            "nodes": nodes_interpolation,
            "edges": edges_interpolation,
            "time": t
        } 

        # apply the model
        nodes_logit, edges_logit_0, edges_logit_1 = self(batch)

        # compute the loss (cross entropy for the edges )
        loss_0 = torch.nn.functional.cross_entropy(edges_logit_0, edges_element[:, :, 0])
        loss_1 = torch.nn.functional.cross_entropy(edges_logit_1, edges_element[:, :, 1])

        loss = loss_0 + loss_1

        self.log("train_loss", loss)

        return loss

    def noise_generation(self, batch_size, num_nodes):
        """
        creating pur noise
        """
        nodes_noise = torch.randn(batch_size, num_nodes)
        edges_noise = torch.randn(batch_size, num_nodes*self.edges_to_node_ratio, 2*num_nodes)

        return nodes_noise, edges_noise
        
    def generation_global(self, batch_size, num_nodes):
        """
        creating pur noise
        """
        nodes_noise, edges_noise = self.noise_generation(batch_size, num_nodes)

        pass

    def configure_optimizers(self):
        """
        Function used to configure the optimizer
        """
        return torch.optim.AdamW(self.parameters(), lr=0.001)

