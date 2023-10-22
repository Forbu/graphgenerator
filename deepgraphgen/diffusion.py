"""
Diffusion model based on https://arxiv.org/pdf/2212.01842.pdf (graphGDP algorithm)

Treat graph as matrices we can compute (as in the paper) a generated graph using the diffusion process


This diffusion model will be used to generate graphs from a given graph

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from deepgraphgen.utils import *

class GraphGDP(nn.Module):
    """
    Implementation of the graphGDP module
    
    This module is composed of two parts : 
        - a first GNN that takes into consideration the full graph
        - a second GNN that takes into consideration the generated graph
    """

