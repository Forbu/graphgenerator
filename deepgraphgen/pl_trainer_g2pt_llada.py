"""
Helper class to train the model (with pytorch lightning)
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.nn import functional as F

import lightning.pytorch as pl

# we create the model
from x_transformers import Encoder

torch.set_float32_matmul_precision("medium")


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
        edges_to_node_ratio=10,
    ):
        super().__init__()

        self.nb_layer = nb_layer
        self.in_dim_node = in_dim_node
        self.out_dim_node = out_dim_node
        self.hidden_dim = hidden_dim
        self.nb_max_node = nb_max_node
        self.edges_to_node_ratio = edges_to_node_ratio

        # transformer core
        self.model_core = Encoder(dim=hidden_dim, depth=nb_layer, heads=8)

        # position embedding
        self.position_embedding = torch.nn.Embedding(
            nb_max_node + edges_to_node_ratio * 2 * nb_max_node, hidden_dim
        )

        # nodes embedding
        self.nodes_embedding = torch.nn.Embedding(nb_max_node + 3, hidden_dim)

        self.epoch_current = 0

    def forward(self, batch):
        # batch["edges"] is (batch_size, num_nodes * self.edges_to_node_ratio * 2)
        batch_size = batch["edges"].shape[0]

        # embedding for nodes (simple vocabulary)
        nodes_embedding_range = torch.arange(self.nb_max_node, device=self.device)

        # we append one token (nb_max_node + 1) at the beggining
        global_embedding = torch.cat(
            [
                nodes_embedding_range.unsqueeze(0).repeat(batch_size, 1).long(),
                torch.ones((batch_size, 1), device=self.device).long()
                * (self.nb_max_node + 1),
                batch["noisy_edges"].long(),
            ],
            dim=1,
        )  # dim is (batch_size, num_nodes + 1 + num_nodes * self.edges_to_node_ratio * 2)

        edges_int = torch.cat(
            [
                torch.ones((batch_size, 1), device=self.device).long()
                * (self.nb_max_node + 1),
                batch["edges"].long(),
            ],
            dim=1,
        )  # dim is (batch_size, 1 + num_nodes * self.edges_to_node_ratio * 2)

        cutting_shape = torch.max((global_embedding != self.nb_max_node).sum(dim=1))
        global_embedding = global_embedding[:, : (cutting_shape + 2)]

        # position integer 0 -> (cutting_shape + 2)
        position_integer = torch.arange(cutting_shape + 2, device=self.device)

        edges_int = edges_int[:, : (cutting_shape - self.nb_max_node + 2)]

        # apply embedding and position embedding AND THEN apply core
        global_embedding = self.nodes_embedding(
            global_embedding
        ) + self.position_embedding(position_integer)

        # apply the transformer core
        output_global = self.model_core(global_embedding)

        edges_output = output_global[:, self.nb_max_node :]

        return edges_int[:, 1:], edges_output[:, :-1, :]

    def training_step(self, batch, batch_idx):
        """
        Function used for the training step
        nodes_element (torch.Tensor): the elements of the nodes (batch_size, num_nodes)
        edges_element (torch.Tensor): the elements of the edges
                                    (batch_size, num_nodes*self.edges_to_node_ratio, 2)
        """
        edges_element = batch[
            "edges"
        ]  # is (batch_size, num_nodes*self.edges_to_node_ratio, 2)

        batch_size = batch["edges"].shape[0]

        # now we want to preprocess edges_element to be
        # (batch_size, num_nodes*self.edges_to_node_ratio* 2)
        # interleave the edges so edges_element_new[:, ::2] = edges_element[:, :, 0]
        # and edges_element_new[:, 1::2] = edges_element[:, :, 1]
        edges_element = edges_element.reshape(
            edges_element.shape[0], edges_element.shape[1] * 2, 1
        ).squeeze(2)

        # mask token randomly
        batch = self.token_masking(edges_element, batch_size)

        # apply the model
        edges_append, edges_logit = self(batch)

        # compute the loss (cross entropy for the edges )
        loss = torch.nn.functional.cross_entropy(
            edges_logit.transpose(1, 2),
            edges_append.long(),
            reduction="mean",
        )

        self.log("train_loss", loss)

        return loss

    def token_masking(self, edges_elements, batch_size):
        # in order to do some discrete diffusion we should do some noise generation
        time_stamp = torch.rand((batch_size), device=self.device)

        batch = {}

        batch["time_stamp"] = time_stamp

        # randomly mask some tokens with a probability of t
        masking = torch.multinomial(
            time_stamp, num_samples=edges_elements.shape, replacement=True
        )
        batch["noisy_edges"] = masking * edges_elements + (1 - masking) * (
            self.nb_max_node + 2
        )
        batch["edges"] = edges_elements

        return batch

    # on the end on epoch
    def on_train_epoch_end(self):
        """
        on the end on epoch
        """
        if self.epoch_current % 10 == 0:
            with torch.no_grad():
                self.generation_global(2)

        self.epoch_current += 1

    def generation_global(self, batch_size):
        """
        creating pur noise
        """

        self.eval()

        # 1. we mask everything first
        # 2. for loop to detokenize and generate the graph
        #    start 1. greedy decoding
        #    start 2. like topk setup (akka autoregressive)
        # 3. plot the graph

        self.plot_graph(output, self.nb_max_node)

        self.train()

    def plot_graph(self, output, nb_max_node):
        """
        Function used to plot the graph
        """

        U = output[0, ::2].long().cpu().numpy()
        V = output[0, 1::2].long().cpu().numpy()

        # if U or V has one element that is nb_max_node we remove it

        G = nx.Graph()
        for i in range(U.shape[0]):
            if U[i] == nb_max_node or V[i] == nb_max_node:
                break
            G.add_edge(U[i], V[i])

        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(10, 10))
        nx.draw(G, pos, with_labels=True)
        plt.savefig("graph_visu/graph_auto.png")

    def configure_optimizers(self):
        """
        Function used to configure the optimizer
        """
        return torch.optim.AdamW(self.parameters(), lr=0.001)
