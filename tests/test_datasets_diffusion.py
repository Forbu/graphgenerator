"""
Test module to test the datasets_diffusion module.
"""

import pytest

from torch_geometric.loader import DataLoader

from deepgraphgen.datasets_diffusion import (
    DatasetGrid,
)


def test_dataset_diffusion_grid():
    # init class for grid graph
    dataset = DatasetGrid(nb_graphs=10, nx=10, ny=10, nb_timestep=1000)

    assert dataset[0]["graph_noisy"].shape == (100, 100)
    assert dataset[0]["gradiant"].shape == (100, 100)

    batch_size = 5

    # now we can test the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        break

    assert batch["graph_noisy"].shape == (batch_size, 100, 100)
    assert batch["gradiant"].shape == (batch_size, 100, 100)
    assert batch["beta"].shape == (batch_size,)
