"""
In this part we will retrieve the datasets use to benchmark graph generation models
"""

from deepgraphgen.datasets_spectre import SpectreGraphDataset

dataset = SpectreGraphDataset(
    dataset_name="planar",
    download_dir="/code/scripts_training/datasets",
    nb_nodes=64,
    edges_to_node_ratio=5,
)

print(dataset[0]["edges"].shape)
print(dataset[0]["nodes"].shape)

# dataloader now
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    print(batch["edges"].shape)
    print(batch["nodes"].shape)
    break