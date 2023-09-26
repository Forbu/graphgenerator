"""
Test module to test the graph data generator
"""

import torch
from torch import nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from deepgraphgen.datasets_torch import DatasetErdos, DatasetGrid


def xtest_dataset_erdos():
    dataset = DatasetErdos(nb_graphs=10, n=100, p=0.01, block_size=10)

    print(dataset[0])

    # try data loader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # try to iterate over the data loader
    for batch in dataloader:
        print(batch)
        print(batch.batch)
        print(batch.block_index)
        # print(batch.edge_imaginary_index)
        print(batch.x)
        print(batch.edge_index)
        print(batch.edge_attr)

        break


def test_dataset_grid():
    dataset = DatasetGrid(nb_graphs=10, nx=10, ny=10, block_size=2)

    # try data loader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # try to iterate over the data loader
    for batch in dataloader:
        print(batch)
        print(batch.batch)
        print(batch.block_index)
        # print(batch.edge_imaginary_index)
        print(batch.edge_index)



        print(batch.edge_attr_imaginary)
        print(batch.edge_attr_imaginary.sum())

        break

    exit()
