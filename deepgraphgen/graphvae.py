"""
Module for GraphVAE

Based on the paper : https://arxiv.org/pdf/1802.03480.pdf

The GraphVAE is a variational autoencoder that learns a latent representation of graphs.
It is composed of an encoder and a decoder. The encoder takes a graph as input and outputs
a latent representation of the graph. The decoder takes a latent representation as input
and outputs a graph.

Encoder is a GNN with a global pooling layer that outputs a latent representation of the graph.
We could also use a GNN with a readout function that outputs a latent representation of the graph.

Decoder is a GNN on a dense graph who output a probability distribution over the edges of the graph.

"""

import torch

class GraphVAE(nn.Module):
    """
    GraphVAE class to compute the latent representation of a graph (encoder)
    and to compute the probability distribution over the edges of the graph (decoder)
    """
    
    def __init__(self, encoder, decoder):
        """
        Constructor of the GraphVAE class
        """
        super(GraphVAE, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, graph):
        """
        Forward pass of the GraphVAE
        """
        latent_representation = self.encoder(graph)
        probability_distribution = self.decoder(latent_representation)
        
        return latent_representation, probability_distribution
    

