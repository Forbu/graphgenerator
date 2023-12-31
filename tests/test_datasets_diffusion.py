"""
Test module to test the datasets_diffusion module.
"""

import pytest

from torch_geometric.loader import DataLoader

from deepgraphgen.datasets_diffusion import (
    DatasetGrid,
    NB_RANDOM_WALK
)


from deepgraphgen.pl_trainer import TrainerGraphGDP


def test_dataset_diffusion_grid():
    # init class for grid graph
    dataset = DatasetGrid(nb_graphs=1000, nx=10, ny=10, nb_timestep=1000)

    assert dataset[0]["graph_noisy"].shape == (100, 100)
    assert dataset[0]["gradiant"].shape == (100, 100)

    batch_size = 128

    # now we can test the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        break

    assert batch["graph_noisy"].shape == (batch_size, 100, 100)
    assert batch["gradiant"].shape == (batch_size, 100, 100)
    assert batch["beta"].shape == (batch_size,)

    model = TrainerGraphGDP(nb_layer=2, hidden_dim=16, nb_max_node=100,
                            dim_edge=NB_RANDOM_WALK+1, dim_node=NB_RANDOM_WALK+1)

    for batch in dataloader:
        model.training_step(batch, 0)
        break
