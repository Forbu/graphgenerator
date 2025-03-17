import os
import pathlib

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch.utils.data import Dataset

from torch_geometric.data import InMemoryDataset


class SpectreGraphDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        download_dir,
        nb_nodes=64,
        edges_to_node_ratio=5,
    ):
        super().__init__()
        self.file_mapping = {
            "sbm": "sbm_200.pt",
            "planar": "planar_64_200.pt",
            "comm20": "community_12_21_100.pt",
        }
        self.dataset_name = dataset_name
        self.download_dir = download_dir
        self.edges_to_node_ratio = edges_to_node_ratio

        # download the dataset
        self.download()

        # preprocess the dataset
        self.preprocess()

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        if self.dataset_name == "sbm":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt"
        elif self.dataset_name == "planar":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt"
        elif self.dataset_name == "comm20":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt"
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

        if os.path.exists(
            os.path.join(self.download_dir, self.file_mapping[self.dataset_name])
        ):
            return

        file_path = download_url(raw_url, self.download_dir)

    def preprocess(self):
        adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = (
            torch.load(
                os.path.join(self.download_dir, self.file_mapping[self.dataset_name])
            )
        )

        for i in range(len(adjs)):
            adjs[i], _ = torch_geometric.utils.dense_to_sparse(adjs[i])

        self.adjs = adjs
        self.n_nodes = n_nodes

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        nb_nodes = self.n_nodes[idx]
        nodes = torch.arange(self.n_nodes[idx])
        edges = self.adjs[idx]

        nb_edges = edges.shape[1]

        # # random permutation on edges index
        # permutation = torch.randperm(nb_nodes)

        # # # random rebal relabel_nodes(G, mapping)
        # edges.cpu().apply_(lambda val: permutation[val])

        if nb_edges < nb_nodes * self.edges_to_node_ratio:
            pad_element = (
                torch.ones(
                    (2, int(nb_nodes * self.edges_to_node_ratio) - nb_edges),
                    dtype=torch.long,
                )
                * self.n_nodes
            )
            edges = torch.cat([edges, pad_element], dim=1)
        elif nb_edges > nb_nodes * self.edges_to_node_ratio:
            edges = edges[:, : int(nb_nodes * self.edges_to_node_ratio)]

        return {
            "nodes": nodes,
            "edges": edges.T,
        }


import os
import os.path as osp
import ssl
import sys
import urllib
from typing import Optional

import fsspec


def download_url(
    url: str,
    folder: str,
    log: bool = True,
    filename: Optional[str] = None,
):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
        filename (str, optional): The filename of the downloaded file. If set
            to :obj:`None`, will correspond to the filename given by the URL.
            (default: :obj:`None`)
    """
    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if os.path.exists(path):  # pragma: no cover
        if log and "pytest" not in sys.modules:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log and "pytest" not in sys.modules:
        print(f"Downloading {url}", file=sys.stderr)

    os.makedirs(folder, exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with fsspec.open(path, "wb") as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


from torch.utils.data import IterableDataset


class DatasetSpectreIterableDataset(IterableDataset):
    def __init__(self, dataset: Dataset):
        super().__init__()  # Initialize the parent class (IterableDataset)
        self.dataset = dataset

    def __iter__(self):
        i = 0
        while True:
            yield self.dataset[
                i % len(self.dataset)
            ]  # Fetch data using the original Dataset's __getitem__
            i += 1
