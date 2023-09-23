"""
Test module to test the graph data generator
"""

import torch
from torch import nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from deepgraphgen.datasets_torch import DatasetErdos

def test_dataset_erdos():
    
    dataset = DatasetErdos(nb_graphs=10, n=100, p=0.01, block_size=10)
    
    print(dataset[0])
    
    # try data loader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # try to iterate over the data loader
    for batch in dataloader:
        print(batch)
        print(batch.batch)
        print(batch.block_index)
        #print(batch.edge_imaginary_index)
        print(batch.x)
        print(batch.edge_index)
        print(batch.edge_attr)
        
        break