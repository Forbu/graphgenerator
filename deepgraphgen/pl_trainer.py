"""
Helper class to train the model (with pytorch lightning)
"""
import networkx as nx
import matplotlib.pyplot as plt

from deepgraphgen.graphGRAN import GRAN
from deepgraphgen.utils import mixture_bernoulli_loss
from deepgraphgen.graphGDP import GraphGDP
from deepgraphgen.datasets_diffusion import (
    compute_mean_value_whole_noise,
    generate_beta_value,
    MAX_BETA,
    MIN_BETA,
)
from deepgraphgen.datasets_diffusion import create_full_graph, create_partial_graph

import torch
import lightning.pytorch as pl
import torchmetrics
import torch_geometric

# we create the model


class TrainerGRAN(pl.LightningModule):
    """
    Warper class for training
    """

    def __init__(
        self,
        nb_layer=2,
        in_dim_node=1,
        out_dim_node=1,
        hidden_dim=16,
        nb_max_node=100,
        dim_order_embedding=16,
        block_size=1,
        nb_k=20,
    ):
        super().__init__()
        self.model = GRAN(
            nb_layer=nb_layer,
            in_dim_node=in_dim_node,
            out_dim_node=out_dim_node,
            hidden_dim=hidden_dim,
            nb_max_node=nb_max_node,
            dim_order_embedding=dim_order_embedding,
            block_size=block_size,
            nb_k=nb_k,
        )

        # init the loss (binary cross entropy with logits)
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        # init accuracy metric
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.train_precision = torchmetrics.Precision(task="binary")

    def forward(self, graphs):
        return self.model(graphs)

    def compute_loss(self, graphs):
        """
        Function used to compute the loss
        """
        _, edges_logit, global_pooling_logit = self.forward(graphs)

        global_pooling_logit_edges_form = torch.index_select(
            global_pooling_logit, 0, graphs.batch
        )

        # also retrieve the edge_attr_imaginary from the graph to compute the loss
        edge_attr_imaginary = graphs.edge_attr_imaginary

        # compute the loss
        loss = mixture_bernoulli_loss(
            edge_attr_imaginary,
            edges_logit,
            global_pooling_logit_edges_form,
            self.loss,
            graphs.batch.unsqueeze(1),
        )
        # self.loss(edges_logit.squeeze(), edge_attr_imaginary.squeeze())

        return loss, edges_logit, edge_attr_imaginary

    def training_step(self, batch, batch_idx):
        """
        Function used for the training step
        """
        loss, edges_prob, edge_attr_imaginary = self.compute_loss(batch)

        # log the loss
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Function used for the validation step
        """
        loss, edges_prob, edge_attr_imaginary = self.compute_loss(batch)

        # log the loss
        self.log("val_loss", loss, batch_size=batch.x.size(0))

        return loss

    def on_validation_epoch_end(self):
        # here we want to sample a graph with the generate() function
        graph = self.model.generate()

        # we can plot the graph
        nx.draw(graph, with_labels=True)

        # save the images (matplotlib) (overwrite) and log it
        name_img = "graph_{}_epoch.png".format(self.current_epoch)
        plt.savefig(name_img)

        # clear the plot
        plt.clf()

        img = plt.imread(name_img)[:, :, :3]

        # change format from HWC to CHW
        img = img.transpose((2, 0, 1))

        self.logger.experiment.add_image(
            "generated_graph", img, self.current_epoch)

    def configure_optimizers(self):
        """
        Function used to configure the optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=0.0001)


# we create the model
class TrainerGraphGDP(pl.LightningModule):
    """
    Warper class for training
    """

    def __init__(self, nb_layer=2, hidden_dim=16, nb_max_node=100):
        super().__init__()
        self.model = GraphGDP(
            nb_layer=nb_layer,
            hidden_dim=hidden_dim,
            nb_max_node=nb_max_node,
        )

        self.model = torch_geometric.compile(self.model, dynamic=True)

        # init the loss (MSE)
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

        # init MSE metric
        self.train_accuracy = torchmetrics.MeanSquaredError()

    def forward(self, graph_1, graph_2, t_value):
        return self.model(graph_1, graph_2, t_value)

    def compute_loss(self, batch):
        """
        Function used to compute the loss
        """
        graph_1 = batch["data_full"]
        graph_2 = batch["data_partial"]
        t_value = batch["timestep"]

        output = self.forward(graph_1, graph_2, t_value)

        # now we compute the loss
        loss = self.loss_fn(output.squeeze().float(),
                            graph_1.edge_attr[:, 1].float())

        return loss

    def training_step(self, batch, batch_idx):
        """
        Function used for the training step
        """
        loss = self.compute_loss(batch)

        # log the loss
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Function used for the validation step
        """
        loss = self.compute_loss(batch)

        # log the loss
        self.log("val_loss", loss, batch_size=batch["data_full"].batch)

        return loss

    def on_validation_epoch_end(self):
        """
        At the end of the epoch we want to generate some graphs
        to check the algorithm global performance
        """

    def configure_optimizers(self):
        """
        Function used to configure the optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def generate(self):
        """
        Reverse diffusion process after the training phase

        Equation :
        At-∆t = At + [1/2 β(t)At + β(t)sθ(At, A¯ t, t)] ∆t + √β(t) √∆t z
        """
        t_array = torch.linspace(0, 1, 1000)
        beta_values = generate_beta_value(MIN_BETA, MAX_BETA, t_array)

        mean_values, variance_values = compute_mean_value_whole_noise(
            t_array, beta_values
        )

        # first we initialize the adjacency matrix with gaussian noise
        # 0 mean and the last variance value
        graph_noisy = torch.randn(
            self.nb_max_node, self.nb_max_node) * variance_values[-1]

        # we create a zero gradient tensor
        gradiant = torch.zeros_like(graph_noisy, requires_grad=False)

        data_full = create_full_graph(graph_noisy, gradiant)
        data_partial = create_partial_graph(graph_noisy)

        delta_t = 0.001

        for idx, time_step in enumerate(t_array[::-1]):
            t_value = torch.tensor(time_step).unsqueeze(0) # should be a tensor of shape (1,)
            output = self.forward(data_full, data_partial, t_value)

            # init the new graph
            s_matrix = torch.zeros_like(graph_noisy)

            # fill with values from edge_attr (output) knowing data_full.edge_index
            s_matrix[data_full.edge_index[0], data_full.edge_index[1]] = output

            beta_current = beta_values[999 - idx]

            # now we can update the graph_noisy according to the equation
            graph_noisy = graph_noisy + beta_current * (0.5  * graph_noisy + s_matrix) * delta_t + \
                        torch.sqrt(beta_current) * torch.sqrt(delta_t) * torch.randn_like(graph_noisy)

            # we update the data_full and data_partial
            data_full = create_full_graph(graph_noisy, gradiant)
            data_partial = create_partial_graph(graph_noisy)

        return graph_noisy


