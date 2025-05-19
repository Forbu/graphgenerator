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

from dit_ml.dit import DiT

torch.set_float32_matmul_precision("medium")

import os


class TrainerG2PT(pl.LightningModule):
    """
    Warper class for training
    """

    def __init__(
        self,
        in_dim_node=1,
        out_dim_node=1,
        hidden_dim=768,
        nb_layer=12,
        heads=12,
        nb_max_node=100,
        edges_to_node_ratio=5,
        step_before_mean_sampling=50,
    ):
        super().__init__()

        self.nb_layer = nb_layer
        self.in_dim_node = in_dim_node
        self.out_dim_node = out_dim_node
        self.hidden_dim = hidden_dim
        self.nb_max_node = nb_max_node
        self.edges_to_node_ratio = edges_to_node_ratio
        self.step_before_mean_sampling = step_before_mean_sampling

        self.nb_edges = edges_to_node_ratio * nb_max_node

        # transformer core
        self.model_core = DiT(
            num_patches=nb_max_node + self.nb_edges,
            hidden_size=hidden_dim,
            depth=nb_layer,
            num_heads=heads,
            dimensionality=1,
        )
        # Encoder(dim=hidden_dim, depth=nb_layer, heads=heads)

        # time projection
        self.time_projection = torch.nn.Linear(1, hidden_dim)

        # position embedding
        self.position_embedding = torch.nn.Embedding(
            nb_max_node + self.nb_edges, hidden_dim
        )

        # nodes embedding
        self.nodes_embedding = torch.nn.Embedding(nb_max_node + 3, hidden_dim // 2)

        # projection toward the output
        self.head_1 = torch.nn.Linear(hidden_dim, self.nb_max_node + 3)
        self.head_2 = torch.nn.Linear(hidden_dim, self.nb_max_node + 3)

        # we construct the Q matrix for masking elements
        self.Q = torch.ones((self.nb_max_node + 3, self.nb_max_node + 3))

        # the value of the diagonal is (1.0 - (self.nb_max_node + 3))
        self.Q.fill_diagonal_(1.0 - (self.nb_max_node + 3))

        self.Q = self.Q / 80.0

        # compute the matrix proper values
        self.L, self.V = torch.linalg.eigh(self.Q)

        self.diag_L = torch.diag(self.L)
        self.inv_V = self.V.T

        # put in parameter Q, L, diag_L, V, inv_V
        self.register_buffer("re_Q", self.Q)
        self.register_buffer("re_L", self.L)
        self.register_buffer("re_diag_L", self.diag_L)
        self.register_buffer("re_V", self.V)
        self.register_buffer("re_inv_V", self.inv_V)

        self.epoch_current = 0

    def forward(self, batch):
        """
        Forward pass
        """
        # batch["edges"] is (batch_size, num_nodes * self.edges_to_node_ratio * 2)
        batch_size = batch["noisy_edges"].shape[0]

        # embedding for nodes (simple vocabulary)
        nodes_embedding_range = torch.arange(self.nb_max_node, device=self.device)

        # batch["noisy_edges"].long()
        noisy_edges = batch["noisy_edges"].reshape(
            batch_size, self.nb_max_node * self.edges_to_node_ratio, 2
        )

        nb_positions = self.nb_max_node * self.edges_to_node_ratio + self.nb_max_node

        # position integer 0 -> (cutting_shape + 2)
        position_integer = torch.arange(nb_positions, device=self.device).unsqueeze(0)
        position_integer = position_integer.repeat(batch_size, 1)

        global_embedding = self.position_embedding(position_integer)

        nodes_embedding_range = self.nodes_embedding(noisy_edges)

        # apply embedding and position embedding AND THEN apply core
        # global_embedding[:, self.nb_max_node :, :] = (
        #     # global_embedding[:, self.nb_max_node :, :]
        #     + nodes_embedding_range[:, :, 0, :]
        #     + nodes_embedding_range[:, :, 1, :]
        # )

        global_embedding[:, self.nb_max_node :, : (self.hidden_dim // 2)] = (
            global_embedding[:, self.nb_max_node :, : (self.hidden_dim // 2)]
            + nodes_embedding_range[:, :, 0, :]
        )

        global_embedding[:, self.nb_max_node :, (self.hidden_dim // 2) :] = (
            global_embedding[:, self.nb_max_node :, (self.hidden_dim // 2) :]
            + nodes_embedding_range[:, :, 1, :]
        )

        time_embedding = self.time_projection(batch["time_stamp"].unsqueeze(1))

        # apply the transformer core
        output_global = self.model_core(global_embedding, time_embedding)

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

        edges_score = F.elu(edges_output)
        edges_score = edges_score + 1.0

        return edges_score

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

        # we want to do the mean field sampling
        batch = self.token_masking(edges_element, batch_size, return_mask=True)

        # breakpoint()

        # apply the model
        edges_score = self(batch)

        loss = (
            edges_score
            - torch.log(edges_score + 1e-8) * batch["proba_y"] / batch["proba_x"]
        )

        loss = loss * batch["mask_x"]
        loss = loss.sum(dim=-1).mean()

        self.log("train_loss", loss)

        if self.global_step % 200 == 0 and self.global_step > 0:
            with torch.no_grad():
                self.generation_global(2, 300)

        return loss

    def token_masking(
        self,
        edges_elements,
        batch_size,
        time_stamp=None,
        return_mask=False,
        stratified=True,
    ):
        with torch.no_grad():
            # in order to do some discrete diffusion we should do some noise generation
            if time_stamp is None:
                if stratified:
                    # stratified sampling
                    time_stamp = stratified_uniform_sample(
                        batch_size, device=self.device
                    )

                else:
                    time_stamp = torch.rand((batch_size), device=self.device)

            batch = {}
            batch["time_stamp"] = time_stamp

            # compute transition matrices
            time_Q = time_stamp.unsqueeze(1) * self.re_L.unsqueeze(0)
            exp_time_Q = torch.exp(time_Q)

            exp_time_Q_emb = torch.diag_embed(exp_time_Q, offset=0, dim1=-2, dim2=-1)

            # now we batch compute the matrix
            full_transition_matrix = torch.bmm(
                self.re_V.unsqueeze(0).repeat(batch_size, 1, 1),
                exp_time_Q_emb,
            )

            full_transition_matrix = torch.bmm(
                full_transition_matrix,
                self.re_inv_V.unsqueeze(0).repeat(batch_size, 1, 1),
            )

            edges_elements_one_hot = torch.nn.functional.one_hot(
                edges_elements, num_classes=self.nb_max_node + 3
            ).transpose(1, 2)
            edges_elements_one_hot = edges_elements_one_hot.float()

            # now we get the probability of the edges
            edges_elements_one_hot_proba = torch.bmm(
                full_transition_matrix,
                edges_elements_one_hot,
            )

            #
            edges_elements_one_hot_proba = edges_elements_one_hot_proba.transpose(1, 2)

            # numerical error
            edges_elements_one_hot_proba = torch.clamp(
                edges_elements_one_hot_proba, min=1e-8, max=1.0
            )

            try:
                # now we want to sample from the distribution
                sampler = torch.distributions.categorical.Categorical(
                    probs=edges_elements_one_hot_proba
                )

                # sampling
                edges_elements_one_hot_sample = sampler.sample()
            except Exception as e:
                print("e", e)
                breakpoint()

            batch["noisy_edges"] = edges_elements_one_hot_sample
            batch["proba_y"] = edges_elements_one_hot_proba

            batch["proba_x"] = torch.gather(
                edges_elements_one_hot_proba,
                2,
                edges_elements_one_hot_sample.unsqueeze(-1),
            )

            mask_x_scatter = torch.ones_like(
                edges_elements_one_hot_proba, dtype=torch.float32
            )
            indices_to_zero = edges_elements_one_hot_sample.unsqueeze(-1)

            mask_x_scatter.scatter_(
                dim=2, index=indices_to_zero, value=0.0
            )  # or src=0.0

            batch["mask_x"] = mask_x_scatter

        return batch

    # on the end on epoch
    def on_train_epoch_end(self):
        """
        on the end on epoch
        """
        if self.epoch_current % 100 == 0:
            with torch.no_grad():
                self.generation_global(1, 300)

        self.epoch_current += 1

    def generation_global(self, batch_size, nb_step):
        """
        creating pur noise
        """

        self.eval()

        # we start from a random sampling
        sampler = torch.distributions.categorical.Categorical(
            probs=torch.ones(
                (
                    batch_size,
                    self.nb_max_node * self.edges_to_node_ratio * 2,
                    self.nb_max_node + 3,
                ),
                device=self.device,
            )
            / (self.nb_max_node + 3)
        )

        edges_elements = sampler.sample()
        edges_elements = edges_elements.long()

        batch = {}
        batch["noisy_edges"] = edges_elements
        batch["time_stamp"] = torch.ones((batch_size), device=self.device) * 1.0

        for i in range(nb_step):
            time = torch.ones((batch_size)) * (1 - i / nb_step)
            time = time.to(self.device)

            score_output = self(batch)

            proba_sample_inverse = (
                self.re_Q[edges_elements, :] * score_output
            )  # size (batch_size, element, num_nodes + 3)

            index_tensor = edges_elements.unsqueeze(-1)

            # we want to correct the distribution
            values_to_write = torch.zeros_like(index_tensor, dtype=score_output.dtype)
            proba_sample_inverse.scatter_(
                dim=2, index=index_tensor, src=values_to_write
            )

            values_to_write_normalization = -torch.sum(
                proba_sample_inverse, dim=2, keepdim=True
            )

            proba_sample_inverse.scatter_(
                dim=2, index=index_tensor, src=values_to_write_normalization
            )

            proba_sample_inverse = 1 / nb_step * proba_sample_inverse

            values_to_add = torch.ones_like(
                index_tensor, dtype=score_output.dtype
            )  # Ensure same dtype

            proba_sample_inverse.scatter_add_(
                dim=2, index=index_tensor, src=values_to_add
            )

            proba_sample_inverse = torch.clamp(proba_sample_inverse, min=1e-8, max=1.0)

            sampler = torch.distributions.categorical.Categorical(
                probs=proba_sample_inverse
            )
            edges_elements = sampler.sample()
            edges_elements = edges_elements.long()
            batch["noisy_edges"] = edges_elements
            batch["time_stamp"] = time

        output = edges_elements
        self.plot_graph(output, self.nb_max_node)

        self.train()

        return None

    def plot_graph(self, output, nb_max_node):
        """
        Function used to plot the graph
        """
        batch_size = output.shape[0]

        nb_planar = 0

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

            # check if graph is planar
            is_planar, _ = nx.check_planarity(G)
            if is_planar:
                nb_planar += 1

            print("Is the graph planar?", is_planar)

            # Set node colors based on the eigenvectors
            w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(G).toarray())
            vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
            m = max(np.abs(vmin), vmax)
            vmin, vmax = -m, m

            plt.figure(figsize=(10, 10))
            nx.draw(
                G,
                pos,
                font_size=5,
                node_size=60,
                with_labels=False,
                node_color=U[:, 1],
                cmap=plt.cm.coolwarm,
                vmin=vmin,
                vmax=vmax,
                edge_color="grey",
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

        print("nb planar graphs proportion", nb_planar / batch_size)

    def configure_optimizers(self):
        """
        Function used to configure the optimizer
        """
        optimizer = ForeachSOAP(
            self.parameters(),
            lr=1e-3,
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


def stratified_uniform_sample(batch_size, device=None, dtype=torch.float32):
    """
    Generates stratified uniform samples between 0 and 1.

    Divides the [0, 1] interval into `batch_size` strata and samples
    one point uniformly from each stratum.

    Args:
        batch_size (int): The number of samples to generate (and the number of strata).
        device (torch.device, optional): The desired device for the tensor.
                                        Defaults to None (CPU).
        dtype (torch.dtype, optional): The desired data type for the tensor.
                                        Defaults to torch.float32.


    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the stratified samples.
                    Each sample t_i is guaranteed to be within the interval
                    [i/batch_size, (i+1)/batch_size).
    """
    if device is None:
        device = torch.device("cpu")  # Default to CPU if no device is specified

    # 1. Create the boundaries for the strata.
    #    We generate points 0, 1, ..., batch_size-1
    t = torch.arange(batch_size, device=device, dtype=dtype)

    # 2. Generate uniform random offsets within [0, 1) for each stratum.
    #    These offsets determine where within each stratum the sample is drawn.
    offsets = torch.rand(batch_size, device=device, dtype=dtype)

    # 3. Calculate the samples.
    #    For each i from 0 to batch_size-1:
    #    - The i-th stratum is [i/batch_size, (i+1)/batch_size).
    #    - We calculate (i + offset_i) / batch_size.
    #    - Since offset_i is in [0, 1), this value falls within the i-th stratum.
    time_stamps = (t + offsets) / batch_size

    return time_stamps
