"""
Different util tools for graph generation
"""

import torch
from torch import nn

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool

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
        assert self.nb_nodes * self.nb_nodes == self.out_dim, "The number of nodes must be equal to the square root of the output dimension"
        
        # Complete modelisation of the decoder
        # basicly a MLP with multiple output : nb_nodes * nb_nodes output for the edges probability
        self.decoder = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, latent_representation):
        
        # Compute the probability distribution over the edges of the graph
        proba = self.decoder(latent_representation)
        
        # Reshape the probability distribution to a matrix
        proba = proba.view(-1, self.nb_nodes, self.nb_nodes)
        
        return proba
        
        
    
