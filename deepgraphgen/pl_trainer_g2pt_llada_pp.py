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

from heavyball import ForeachSOAP, ForeachMuon

torch.set_float32_matmul_precision("medium")

import os


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
        edges_to_node_ratio=5,
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

        self.nb_edges = edges_to_node_ratio * nb_max_node

        # position embedding
        self.position_embedding = torch.nn.Embedding(
            nb_max_node + self.nb_edges, hidden_dim
        )

        # nodes embedding
        self.nodes_embedding = torch.nn.Embedding(nb_max_node + 3, hidden_dim)

        # projection toward the output
        self.head_1 = torch.nn.Linear(hidden_dim, self.nb_max_node + 3)
        self.head_2 = torch.nn.Linear(hidden_dim, self.nb_max_node + 3)

        self.epoch_current = 0

    def forward(self, batch):
        """
        Forward pass
        """
        # batch["edges"] is (batch_size, num_nodes * self.edges_to_node_ratio * 2)
        batch_size = batch["edges"].shape[0]

        # embedding for nodes (simple vocabulary)
        nodes_embedding_range = torch.arange(self.nb_max_node, device=self.device)

        # batch["noisy_edges"].long()
        noisy_edges = batch["noisy_edges"].reshape(
            batch_size, self.nb_max_node * self.edges_to_node_ratio, 2
        )

        edges_int = batch["edges"].long()

        nb_positions = self.nb_max_node * self.edges_to_node_ratio + self.nb_max_node

        # position integer 0 -> (cutting_shape + 2)
        position_integer = torch.arange(nb_positions, device=self.device).unsqueeze(0)
        position_integer = position_integer.repeat(batch_size, 1)

        global_embedding = self.position_embedding(position_integer)

        nodes_embedding_range = self.nodes_embedding(noisy_edges)

        # apply embedding and position embedding AND THEN apply core
        global_embedding[:, self.nb_max_node :, :] = (
            #global_embedding[:, self.nb_max_node :, :]
            + nodes_embedding_range[:, :, 0, :]
            + nodes_embedding_range[:, :, 1, :]
        )

        # apply the transformer core
        output_global = self.model_core(global_embedding)

        edges_output = output_global[:, self.nb_max_node :]

        # apply the projection
        edges_output_1 = self.head_1(edges_output)
        edges_output_2 = self.head_2(edges_output)

        # now we concatenate
        edges_output = torch.stack([edges_output_1, edges_output_2], dim=-1).reshape(
            edges_output.shape[0],
            self.nb_max_node * self.edges_to_node_ratio * 2,
            self.nb_max_node + 3,
        )

        return edges_int, edges_output

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
        batch, mask = self.token_masking(edges_element, batch_size, return_mask=True)

        # breakpoint()

        # apply the model
        edges_append, edges_logit = self(batch)

        # compute the loss (cross entropy for the edges )
        loss = torch.nn.functional.cross_entropy(
            edges_logit.transpose(1, 2),
            edges_append.long(),
            reduction="none",
        )

        # compute the loss for the masked tokens
        loss = loss * mask.float()
        loss = loss.sum() / mask.sum()

        self.log("train_loss", loss)

        if self.global_step % 200 == 0:
            with torch.no_grad():
                self.generation_global(2, 300, remasking="low_entropy")

        return loss

    def token_masking(
        self, edges_elements, batch_size, time_stamp=None, return_mask=False
    ):
        # in order to do some discrete diffusion we should do some noise generation
        if time_stamp is None:
            time_stamp = torch.rand((batch_size), device=self.device)

        batch = {}
        batch["time_stamp"] = time_stamp

        # randomly mask some tokens with a probability of t
        proba_compute = torch.cat(
            [time_stamp.unsqueeze(1), 1 - time_stamp.unsqueeze(1)], dim=1
        )
        masking = torch.multinomial(
            proba_compute, num_samples=edges_elements.shape[1], replacement=True
        )

        batch["noisy_edges"] = (1 - masking) * edges_elements + masking * (
            self.nb_max_node + 2
        )

        batch["noisy_edges"] = batch["noisy_edges"].long()

        batch["edges"] = edges_elements

        if return_mask:
            return batch, masking
        return batch

    # on the end on epoch
    def on_train_epoch_end(self):
        """
        on the end on epoch
        """
        if self.epoch_current % 100 == 0:
            with torch.no_grad():
                self.generation_global(2, 300, remasking="low_entropy")

        self.epoch_current += 1

    def generation_global(self, batch_size, nb_step, remasking="low_confidence"):
        """
        creating pur noise
        """

        self.eval()

        # 1. we mask everything first
        time_stamp = torch.zeros((batch_size), device=self.device)  # full masking
        edges_element = torch.ones(
            (batch_size, self.nb_max_node * self.edges_to_node_ratio * 2),
            device=self.device,
        )

        batch = self.token_masking(edges_element, batch_size, time_stamp)

        delta_choose = (self.nb_max_node * self.edges_to_node_ratio * 2) // nb_step

        print("begin generation")

        # 2. for loop to detokenize and generate the graph
        for i in range(nb_step):
            # get logits prediction
            # print("begin step", i)

            edges_append, edges_logit = self(batch)

            # edges logit are dim (batch_size, num_nodes*self.edges_to_node_ratio, vocab_size)
            # we put all the edges logit from non-mask token at -1000.
            non_mask_token = batch["noisy_edges"] != (self.nb_max_node + 2)

            delta_choose = (~non_mask_token).sum(dim=1)[0] // (
                nb_step - i
            )  # number of tokens to choose

            # sample from softmax
            softmax_p = torch.nn.functional.softmax(edges_logit, dim=2)

            # sampling from softmax
            max_proba_index = torch.multinomial(softmax_p.flatten(0, 1), num_samples=1)

            max_proba_index = max_proba_index.reshape(
                batch_size, self.nb_max_node * self.edges_to_node_ratio * 2
            )

            if remasking == "low_confidence":
                max_logit = torch.max(softmax_p, dim=2)[
                    0
                ]  # dim is (batch_size, num_nodes*self.edges_to_node_ratio)

                # we don't choose the mask token
                max_logit = torch.where(
                    non_mask_token, torch.full_like(max_logit, -1000), max_logit
                )

                # now we want to retrieve the topk values
                _, indices = torch.topk(max_logit, k=delta_choose, dim=1)

            elif remasking == "low_entropy":
                # we compute the entropy value for each token
                entropy = torch.sum(softmax_p * torch.log(softmax_p), dim=2)

                # we remove the mask token
                entropy = torch.where(
                    non_mask_token,
                    torch.full_like(entropy, -1000),
                    entropy,
                )

                # now we want to retrieve the topk values
                _, indices = torch.topk(entropy, k=delta_choose, dim=1)

            new_noisy_value = batch["noisy_edges"].clone()

            # create the indice map
            index_batch = (
                torch.arange(batch_size, device=self.device)
                .unsqueeze(1)
                .repeat(1, delta_choose)
            )

            # update the noisy value
            new_noisy_value[index_batch.flatten().long(), indices.flatten().long()] = (
                max_proba_index[index_batch.flatten().long(), indices.flatten().long()]
            )

            # replace the noisy value with the new value
            batch["noisy_edges"] = new_noisy_value

        print("end generation")

        output = batch["noisy_edges"].long()

        self.plot_graph(output, self.nb_max_node)

        self.train()

    def plot_graph(self, output, nb_max_node):
        """
        Function used to plot the graph
        """
        batch_size = output.shape[0]

        for batch_idx in range(batch_size):
            elements = set()

            U = output[batch_idx, ::2].long().cpu().numpy()
            V = output[batch_idx, 1::2].long().cpu().numpy()

            # if U or V has one element that is nb_max_node we remove it

            G = nx.Graph()
            for i in range(U.shape[0]):
                if U[i] >= nb_max_node or V[i] >= nb_max_node:
                    pass
                else:
                    G.add_edge(U[i], V[i])

                    # append U[i] and V[i] to the set
                    elements.add(U[i])
                    elements.add(V[i])

            pos = nx.spring_layout(G, seed=42)

            plt.figure(figsize=(10, 10))
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color=range(len(elements)),
                cmap=plt.cm.viridis,
            )

            name_img = (
                "graph_visu/graph_auto_epoch_"
                + str(self.epoch_current)
                + "_"
                + str(batch_idx)
                + ".png"
            )

            plt.savefig(name_img)

            # now we clean the plot
            plt.clf()

            # now we log the png image (tensorboard)
            img = plt.imread(name_img)[:, :, :3]
            img = img.transpose((2, 0, 1))

            # we log the figure to tensorboard
            self.logger.experiment.add_image(
                "pred_image", img, global_step=self.global_step
            )

            # remove the image
            os.remove(name_img)

    def configure_optimizers(self):
        """
        Function used to configure the optimizer
        """
        optimizer = ForeachMuon(
            self.parameters(),
            lr=1e-3,
            betas=(0.95, 0.95),
            weight_decay=1e-1,
            foreach=False,
            warmup_steps=1000,
        )

        # optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)

        return optimizer


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
