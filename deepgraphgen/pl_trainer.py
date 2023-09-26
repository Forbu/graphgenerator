"""
Helper class to train the model (with pytorch lightning)
"""


from deepgraphgen.graphGRAN import GRAN

import torch
import lightning.pytorch as pl

import torchmetrics


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
    ):
        super().__init__()
        self.model = GRAN(
            nb_layer=nb_layer,
            in_dim_node=in_dim_node,
            out_dim_node=out_dim_node,
            hidden_dim=hidden_dim,
            nb_max_node=nb_max_node,
            dim_order_embedding=dim_order_embedding,
        )

        # init the loss (binary cross entropy)
        self.loss = torch.nn.BCELoss()

        # init accuracy metric
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.train_precision = torchmetrics.Precision(task="binary")

    def forward(self, graphs):
        return self.model(graphs)

    def compute_loss(self, graphs):
        """
        Function used to compute the loss
        """
        _, edges_prob = self.forward(graphs)

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
        self.log(
            "train_accuracy",
            self.train_accuracy(edges_prob.squeeze(), edge_attr_imaginary.squeeze())
            .cpu()
            .item(),
        )

        # log precision between the predicted and the real edge
        self.log(
            "train_precision",
            self.train_precision(edges_prob.squeeze(), edge_attr_imaginary.squeeze())
            .cpu()
            .item(),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Function used for the validation step
        """
        loss, edges_prob, edge_attr_imaginary = self.compute_loss(batch)

        # log the loss
        self.log("val_loss", loss)

        # log accuracy between the predicted and the real edge
        self.log(
            "val_accuracy",
            self.train_accuracy(edges_prob.squeeze(), edge_attr_imaginary.squeeze())
            .cpu()
            .item(),
        )

        # log precision between the predicted and the real edge
        self.log(
            "val_precision",
            self.train_precision(edges_prob.squeeze(), edge_attr_imaginary.squeeze())
            .cpu()
            .item(),
        )

        return loss

    def configure_optimizers(self):
        """
        Function used to configure the optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=0.0001)
