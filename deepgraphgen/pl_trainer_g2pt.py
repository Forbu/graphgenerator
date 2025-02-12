"""
Helper class to train the model (with pytorch lightning)
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.nn import functional as F

import lightning.pytorch as pl

import matplotlib.pyplot as plt

# we create the model
from x_transformers import Encoder

torch.set_float32_matmul_precision('medium')

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
        self.model_core = Encoder(dim=hidden_dim, depth=6, heads=8)

        # embedding for the ndoes
        self.nodes_embedding = torch.nn.Embedding(nb_max_node, hidden_dim)

        # linear layer for the edgesinput
        self.edges_embedding = torch.nn.Sequential(
            torch.nn.Linear(2 * (nb_max_node + 1), hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        self.edges_embedding_position = torch.nn.Embedding(
            nb_max_node * self.edges_to_node_ratio, hidden_dim
        )

        # linear layer for the time
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        # last layers for nodes edges 0 and 1
        self.nodes_logit = torch.nn.Linear(hidden_dim, nb_max_node)
        self.edges_logit_0 = torch.nn.Linear(hidden_dim, nb_max_node + 1)
        self.edges_logit_1 = torch.nn.Linear(hidden_dim, nb_max_node + 1)

        #
        self.epoch_current = 0

    def forward(self, batch):
        batch_size, nb_max_node = (
            batch["edges"].shape[0],
            batch["edges"].shape[1] // self.edges_to_node_ratio,
        )
        # apply the embedding
        nodes_embedding_range = torch.arange(self.nb_max_node, device=self.device)
        nodes_embedding_range = (
            nodes_embedding_range.unsqueeze(0).repeat(batch_size, 1).long()
        )

        edges_embedding_range = torch.arange(
            nb_max_node * self.edges_to_node_ratio, device=self.device
        )
        edges_embedding_range = (
            edges_embedding_range.unsqueeze(0).repeat(batch_size, 1).long()
        )

        nodes_embedding = self.nodes_embedding(nodes_embedding_range)
        edges_embedding_position = self.edges_embedding_position(edges_embedding_range)

        edges_embedding = self.edges_embedding(batch["edges"]) + edges_embedding_position
        time_embedding = self.time_embedding(batch["time"])

        global_embedding = torch.cat([nodes_embedding, edges_embedding], dim=1)

        # apply film layer with time embedding
        global_embedding = global_embedding * (1.0 + time_embedding)

        # apply the transformer core
        output_global = self.model_core(global_embedding)

        # angain we apply film layer with time embedding
        output_global = output_global * (1.0 + time_embedding)

        nodes_output = output_global[:, : self.nb_max_node]
        edges_output = output_global[:, self.nb_max_node :]

        # apply the embedding
        nodes_logit = self.nodes_logit(nodes_output)
        edges_logit_0 = self.edges_logit_0(edges_output)
        edges_logit_1 = self.edges_logit_1(edges_output)

        return nodes_logit, edges_logit_0, edges_logit_1

    def training_step(self, batch, batch_idx):
        """
        Function used for the training step
        nodes_element (torch.Tensor): the elements of the nodes (batch_size, num_nodes)
        edges_element (torch.Tensor): the elements of the edges
                                    (batch_size, num_nodes*self.edges_to_node_ratio, 2)
        """
        nodes_element = batch["nodes"]
        edges_element = batch["edges"]

        num_nodes = nodes_element.shape[1]
        batch_size = nodes_element.shape[0]

        nodes_noise, edges_noise = self.noise_generation(batch_size, num_nodes)

        # convert nodes element and edges element to one hot encoding
        nodes_element = torch.nn.functional.one_hot(
            nodes_element, num_classes=num_nodes
        )
        edges_element_onehot = torch.nn.functional.one_hot(
            edges_element, num_classes=num_nodes + 1
        )  # shape (batch_size, num_nodes*self.edges_to_node_ratio, 2, num_nodes + 1)

        # flatten the edges element
        edges_element_onehot = edges_element_onehot.reshape(
            batch_size, num_nodes * self.edges_to_node_ratio, 2 * (num_nodes + 1)
        )

        # generate the random noise
        t = torch.rand((batch_size, 1, 1), device=self.device)

        # generate the interpolation
        nodes_interpolation = t * nodes_element + (1 - t) * nodes_noise
        edges_interpolation = t * edges_element_onehot + (1 - t) * edges_noise

        batch = {"edges": edges_interpolation, "time": t}

        # apply the model
        nodes_logit, edges_logit_0, edges_logit_1 = self(batch)

        coef = t / (1.0 - t)
        coef = coef.clamp(min=0.005, max=1.5)

        # compute the loss (cross entropy for the edges )
        loss_0 = torch.nn.functional.cross_entropy(
            edges_logit_0.transpose(1, 2), edges_element[:, :, 0].long(), reduction="none"
        )
        loss_1 = torch.nn.functional.cross_entropy(
            edges_logit_1.transpose(1, 2), edges_element[:, :, 1].long(), reduction="none"
        )

        loss_0 = (coef * loss_0).mean()
        loss_1 = (coef * loss_1).mean()

        loss = loss_0 + loss_1

        self.log("train_loss", loss)

        return loss

    def noise_generation(self, batch_size, num_nodes):
        """
        creating pur noise
        """
        nodes_noise = torch.randn(batch_size, num_nodes, num_nodes, device=self.device)
        edges_noise = torch.randn(
            batch_size, num_nodes * self.edges_to_node_ratio, 2 * (num_nodes + 1), device=self.device
        )

        return nodes_noise, edges_noise

    # on the end on epoch
    def on_train_epoch_end(self):
        """
        on the end on epoch
        """
        if self.epoch_current % 50 == 0:
            with torch.no_grad():
                self.generation_global(2, 100)

        self.epoch_current += 1

    def generation_global(self, batch_size, num_nodes):
        """
        creating pur noise
        """

        self.eval()
        nodes_noise, edges_prior = self.noise_generation(batch_size, num_nodes)

        nb_step = 100

        time_step = torch.linspace(0, 1, nb_step, device=self.device)
        time_step = (
            time_step.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        )

        for i in range(nb_step - 1):
            t = time_step[:, :, :, i]

            batch = {
                "edges": edges_prior,
                "time": t,
            }

            nodes_logit, edges_logit_0, edges_logit_1 = self(batch)

            edges_proba_prior_0 = F.softmax(edges_logit_0, dim=-1)
            edges_proba_prior_1 = F.softmax(edges_logit_1, dim=-1)

            speed_0 = (
                1.0
                / (1.0 - t)
                * (edges_proba_prior_0 - edges_prior[:, :, : (num_nodes + 1)])
            )
            speed_1 = (
                1.0
                / (1.0 - t)
                * (edges_proba_prior_1 - edges_prior[:, :, (num_nodes + 1) :])
            )

            edges_proba_prior_0 += speed_0 * 1.0 / nb_step
            edges_proba_prior_1 += speed_1 * 1.0 / nb_step

            edges_prior = torch.cat(
                [
                    edges_proba_prior_0,
                    edges_proba_prior_1,
                ],
                dim=-1,
            )

        indice_edges_prior_0 = torch.argmax(edges_prior[:, :, :(num_nodes + 1)], dim=-1)
        indice_edges_prior_1 = torch.argmax(edges_prior[:, :, (num_nodes + 1) :], dim=-1)

        # nowe we can build the graph in networkx and draw it with matplotlib
        for i in range(batch_size):
            G = nx.Graph()
            for j in range(num_nodes):
                G.add_node(j)
            for j in range(num_nodes * self.edges_to_node_ratio):
                if (
                    indice_edges_prior_0[i, j] != num_nodes
                    and indice_edges_prior_1[i, j] != num_nodes
                ):
                    G.add_edge(indice_edges_prior_0[i, j].item(), indice_edges_prior_1[i, j].item())
            plt.figure(figsize=(10, 10))
            nx.draw(G, with_labels=True)
            plt.savefig(f"graph_visu/graph_{i}.png")
            plt.close()

        self.train()

    def configure_optimizers(self):
        """
        Function used to configure the optimizer
        """
        return torch.optim.AdamW(self.parameters(), lr=0.001)
