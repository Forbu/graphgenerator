"""
Helper class to train the model (with pytorch lightning)
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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
from deepgraphgen.diffusion_generation import transform_to_symetric

import torch
import lightning.pytorch as pl
import torchmetrics
import torch_geometric

import matplotlib.pyplot as plt

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

        self.logger.experiment.add_image("generated_graph", img, self.current_epoch)

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

    def __init__(
        self,
        nb_layer=2,
        hidden_dim=16,
        nb_max_node=100,
        dim_node=1,
        dim_edge=1,
    ):
        super().__init__()
        self.model = GraphGDP(
            nb_layer=nb_layer,
            hidden_dim=hidden_dim,
            nb_max_node=nb_max_node,
            dim_node=dim_node,
            dim_edge=dim_edge,
        )

        self.nb_max_node = nb_max_node

        # self.model = torch_geometric.compile(self.model, dynamic=True)

        # init the loss (MSE)
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

        # init MSE metric
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()

        # variable for generation
        self.t_array = torch.linspace(0, 1, 1000)
        self.beta_values = generate_beta_value(MIN_BETA, MAX_BETA, self.t_array)

        self.mean_values, self.variance_values = compute_mean_value_whole_noise(
            self.t_array, self.beta_values
        )

    def forward(self, graph_1, graph_2, t_value):
        return self.model(graph_1, graph_2, t_value)

    def compute_weight_loss(self, graph_1, t_value):
        # compute the weight
        subgraph_idx = graph_1.batch

        # create the time encoding
        t_array_nodes = (
            torch.index_select(t_value, 0, subgraph_idx.to(graph_1.x.device))
            .unsqueeze(1)
            .to(graph_1.x.device)
        ).detach()

        t_array_edges = t_array_nodes[graph_1.edge_index[0]]

        index_timestep = torch.floor(t_array_edges * 1000).long()
        index_timestep = torch.clamp(index_timestep, max=999).long()

        tensor_variance = torch.tensor(self.variance_values)

        return tensor_variance.to(graph_1.x.device)[index_timestep].squeeze()

    def compute_loss(self, batch, type="train"):
        """
        Function used to compute the loss
        """
        graph_1 = batch["data_full"]
        graph_2 = batch["data_partial"]
        t_value = batch["timestep"]

        output = self.forward(graph_1, graph_2, t_value)

        def weighted_mse_loss(input, target, weight):
            return torch.mean(weight * (input - target) ** 2)

        with torch.no_grad():
            weight_loss = self.compute_weight_loss(graph_1, t_value)

        # now we compute the loss
        loss = weighted_mse_loss(
            output.squeeze().float(), graph_1.edge_attr[:, 1].float(), weight_loss
        )
        # loss = self.loss_fn(output.squeeze().float(), graph_1.edge_attr[:, 1].float())

        # compute the MSE
        if type == "train":
            mse = self.train_mse(
                output.squeeze().float(), graph_1.edge_attr[:, 1].float()
            )
        else:
            mse = self.val_mse(
                output.squeeze().float(), graph_1.edge_attr[:, 1].float()
            )

        return loss

    def training_step(self, batch, batch_idx):
        """
        Function used for the training step
        """
        loss = self.compute_loss(batch, type="train")

        # log the loss
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Function used for the validation step
        """
        loss = self.compute_loss(batch, type="val")

        # log the loss
        self.log("val_loss", loss)

        return loss

    def on_train_epoch_end(self):
        # we log the mse value
        self.log("train_mse", self.train_mse.compute())

        # we reset the metric
        self.train_mse.reset()

    def on_validation_epoch_end(self):
        """
        At the end of the epoch we want to generate some graphs
        to check the algorithm global performance
        """
        self.eval()
        with torch.no_grad():
            exemples_graphs = self.generate()

        for idx, example_graph in enumerate(exemples_graphs):
            # create an png image
            plt.imshow(example_graph, vmin=-2, vmax=2)

            # add colorbar
            plt.colorbar()

            # save the images (matplotlib) (overwrite) and log it
            name_img = "graph_{}_epoch.png".format(self.current_epoch)

            plt.savefig(name_img)

            # clear the plot
            plt.clf()

            img = plt.imread(name_img)[:, :, :3]

            # log the matrix (100x100) as an image
            if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
                import wandb

                self.logger.experiment.log(
                    {"generated_graph_{}".format(idx): wandb.Image(img)}
                )
            else:
                # change format from HWC to CHW
                img = img.transpose((2, 0, 1))
                self.logger.experiment.add_image(
                    "generated_graph_{}".format(idx), img, self.current_epoch
                )

            # now we also want to transform the graph into a networkx graph to plot it
            adjacency_matrix = example_graph >= 0.5

            # create the graph
            graph = nx.from_numpy_array(adjacency_matrix)

            # plot the graph
            nx.draw(graph, with_labels=True)

            # save the images (matplotlib) (overwrite) and log it
            name_img = "graph_{}_epoch.png".format(self.current_epoch)

            plt.savefig(name_img)

            # clear the plot
            plt.clf()

            img = plt.imread(name_img)[:, :, :3]

            # log the matrix (100x100) as an image
            if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
                import wandb

                self.logger.experiment.log(
                    {"generated_graph_{}_networkx".format(idx): wandb.Image(img)}
                )
            else:
                # change format from HWC to CHW
                img = img.transpose((2, 0, 1))
                self.logger.experiment.add_image(
                    "generated_graph_{}_networkx".format(idx), img, self.current_epoch
                )

        # we also log the mse value
        self.log("val_mse", self.val_mse.compute())

        # we reset the metric
        self.val_mse.reset()

    def configure_optimizers(self):
        """
        Function used to configure the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        def lr_foo(batch_step):
            # every 7000 steps we divide the learning rate by 2
            nb_step = 7000
            lr_scale = 0.8 ** (batch_step // nb_step)

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def generate(self):
        """
        Reverse diffusion process after the training phase

        Equation :
        At-∆t = At + [1/2 β(t)At + β(t)sθ(At, A¯ t, t)] ∆t + √β(t) √∆t z
        """
        # here we get the device
        device = self.device

        nb_node = self.nb_max_node

        # first we initialize the adjacency matrix with gaussian noise
        # 0 mean and the last variance value
        graph_noisy = torch.randn(nb_node, nb_node).to(device) * torch.sqrt(
            torch.tensor(self.variance_values[-1])
        ).to(device)

        graph_noisy = transform_to_symetric(graph_noisy.cpu().numpy())
        graph_noisy = torch.from_numpy(graph_noisy).float().to(device)

        # we create a zero gradient tensor
        gradiant = torch.zeros_like(graph_noisy, requires_grad=False).to(device)

        data_full = create_full_graph(graph_noisy, gradiant)
        data_partial = create_partial_graph(graph_noisy)

        data_full.batch = torch.zeros_like(data_full.x).long().to(device)
        data_partial.batch = torch.zeros_like(data_partial.x).long().to(device)[:, 0]

        delta_t = 0.001

        register_step = [0, 250, 500, 750, 999]

        images_register = []

        for idx, time_step in enumerate(torch.flip(self.t_array, dims=[0])):
            t_value = (
                torch.tensor(time_step).unsqueeze(0).to(device)
            )  # should be a tensor of shape (1,)

            output = self.forward(data_full, data_partial, t_value)

            # init the new graph
            s_matrix = torch.zeros_like(graph_noisy).to(device)

            # fill with values from edge_attr (output) knowing data_full.edge_index
            s_matrix[
                data_full.edge_index[0], data_full.edge_index[1]
            ] = output.squeeze()

            s_matrix = torch.triu(s_matrix) + torch.triu(s_matrix).T
            s_matrix[range(nb_node), range(nb_node)] = (
                s_matrix[range(nb_node), range(nb_node)] / 2
            )

            beta_current = self.beta_values[999 - idx]

            symetric_noise = torch.randn_like(graph_noisy).to(device)
            symetric_noise = transform_to_symetric(symetric_noise.cpu().numpy())
            symetric_noise = torch.from_numpy(symetric_noise).float().to(device)

            # now we can update the graph_noisy according to the equation
            graph_noisy = (
                graph_noisy
                + beta_current * (0.5 * graph_noisy + s_matrix) * delta_t
                + torch.sqrt(torch.tensor(beta_current))
                * torch.sqrt(torch.tensor(delta_t))
                * symetric_noise
            )

            # we update the data_full and data_partial
            data_full = create_full_graph(graph_noisy, gradiant)
            data_partial = create_partial_graph(graph_noisy)

            data_full.batch = torch.zeros_like(data_full.x).long().to(device)
            data_partial.batch = (
                torch.zeros_like(data_partial.x).long().to(device)[:, 0]
            )

            if idx in register_step:
                images_register.append(graph_noisy.cpu().detach().numpy())

        return images_register
