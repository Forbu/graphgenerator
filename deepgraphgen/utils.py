"""
Different util tools for graph generation
"""

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool


def mixture_bernoulli_loss(
    label,
    theta_logits,
    log_alpha,
    adj_loss_func,
    subgraph_idx,
    reduction="mean",
):
    """
    Compute likelihood for mixture of Bernoulli model

    Args:
      label: E X 1, see comments above
      theta_logits: E X D, see comments above
      log_alpha: E X D, see comments above
      adj_loss_func: BCE loss with logits
      subgraph_idx: E X 1, see comments above
      reduction: string, type of reduction on batch dimension ("mean", "sum", "none")

    Returns:
      loss (and potentially neg log prob)
    """

    num_subgraph = subgraph_idx.max() + 1
    E = theta_logits.shape[0]
    K = theta_logits.shape[1]

    adj_loss = torch.stack(
        [adj_loss_func(theta_logits[:, kk], label) for kk in range(K)], dim=1
    )  # E, K, adj loss for each edge

    const = torch.zeros(num_subgraph).to(label.device).unsqueeze(1)  # S

    const = const.scatter_add_(
        0, subgraph_idx, torch.ones_like(subgraph_idx).float()
    )  # nb edges per subgraph

    reduce_adj_loss = torch.zeros((num_subgraph, K)).to(label.device)
    reduce_adj_loss = reduce_adj_loss.scatter_add_(
        0, subgraph_idx.expand(-1, K), adj_loss
    )  # S, K, sum of adj losses for each subgraph

    reduce_log_alpha = torch.zeros((num_subgraph, K)).to(label.device)
    reduce_log_alpha = reduce_log_alpha.scatter_add_(
        0, subgraph_idx.expand(-1, K), log_alpha
    )  # S, K, sum of log alpha for each subgraph

    reduce_log_alpha = reduce_log_alpha / const.view(
        -1, 1
    )  # S, K, average log alpha for each subgraph
    reduce_log_alpha = F.log_softmax(
        reduce_log_alpha, -1
    )  # S, K, log softmax of average log alpha for each subgraph

    log_prob = -reduce_adj_loss + reduce_log_alpha  # S, K, log prob of each subgraph
    log_prob = torch.logsumexp(log_prob, dim=1)  # S, log prob of each subgraph

    if reduction == "mean":
        loss = log_prob.mean()
    elif reduction == "sum":
        loss = log_prob.sum()

    return -loss


class MLP(nn.Module):
    """
    Simple MLP (multi-layer perceptron)
    """

    # MLP with LayerNorm
    def __init__(
        self,
        in_dim,
        out_dim=128,
        hidden_dim=128,
        hidden_layers=2,
        norm_type=None,
    ):
        """
        MLP
        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        normalize_output: if True, normalize output
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', ...
        """

        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm1d",
                "MessageNorm",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, vector):
        """
        Simple forward pass
        """
        return self.model(vector)


class GraphEncoder(nn.Module):
    """
    GraphEncoder class to compute the latent representation of a graph
    Basicly a simple GNN with a global pooling layer
    """

    def __init__(self, nb_layers, in_dim, hidden_dim, out_dim):
        """
        Constructor of the GraphEncoder class
        """
        super(GraphEncoder, self).__init__()

        self.nb_layers = nb_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.gnn = nn.ModuleList()

        for i in range(self.nb_layers):
            if i == 0:
                self.gnn.append(GATv2Conv(self.in_dim, self.hidden_dim))
            elif i == self.nb_layers - 1:
                self.gnn.append(GATv2Conv(self.hidden_dim, self.out_dim))
            else:
                self.gnn.append(GATv2Conv(self.hidden_dim, self.hidden_dim))

    def forward(self, graph):
        """
        Forward pass of the GraphEncoder
        """
        x, edge_index = graph.x, graph.edge_index

        for i in range(self.nb_layers):
            x = self.gnn[i](x, edge_index)

        latent_representation = global_mean_pool(x, graph.batch)

        return latent_representation


class GraphDecoder(nn.Module):
    """
    Class use for the decoding of a graph
    from a latent representation to a probability distribution over the edges of the graph (basicly a image generation)
    """

    def __init__(self, nb_layers, in_dim, hidden_dim, out_dim, nb_nodes):
        """
        Constructor of the GraphDecoder class
        """
        super(GraphDecoder, self).__init__()

        self.nb_layers = nb_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.nb_nodes = nb_nodes
        assert self.nb_nodes > 0, "The number of nodes must be greater than 0"
        assert (
            self.nb_nodes * self.nb_nodes == self.out_dim
        ), "The number of nodes must be equal to the square root of the output dimension"

        # Complete modelisation of the decoder
        # basicly a MLP with multiple output : nb_nodes * nb_nodes output for the edges probability
        self.decoder = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.Sigmoid(),
        )

    def forward(self, latent_representation):
        # Compute the probability distribution over the edges of the graph
        proba = self.decoder(latent_representation)

        # Reshape the probability distribution to a matrix
        proba = proba.view(-1, self.nb_nodes, self.nb_nodes)

        return proba
