import numpy as np
from typing import Tuple, Union
import torch
from torch.utils.data import Dataset
import pandas as pd


class MSA_Dataset(Dataset):
    """
    Custom PyTorch Dataset for holding each sequence in a MSA.

    Before loading values, it will send data to the specified device.

    Inputs:
    encodings: The one-hot encodings of the sequences
    weights: The reweightings of the sequences.
    ids: The sequence id names
    device: The device to send data to.

    Returns:
    encodings, weights, ids
    """

    def __init__(
        self,
        encodings: np.ndarray,
        weights: np.ndarray,
        ids: np.ndarray,
        device: torch.device,
    ) -> None:

        assert encodings.shape[0] == len(weights)
        assert encodings.shape[0] == len(ids)

        self.encodings = torch.tensor(np.stack(encodings)).float().to(device)
        self.weights = torch.tensor(weights).float().to(device)

        self.ids = ids.values

    def __len__(self):
        return self.encodings.shape[0]

    def __getitem__(self, index) -> Tuple[torch.tensor, float, str]:
        return self.encodings[index, :], self.weights[index], self.ids[index]


class DMS_Dataset(Dataset):
    """Holds data for a deep mutational scanning dataset"""

    def __init__(
        self,
        encodings: np.ndarray,
        ids: np.ndarray,
        fitness: np.ndarray,
        fitness_bin: np.ndarray,
        device: torch.device,
    ) -> None:

        assert encodings.shape[0] == len(ids)
        assert encodings.shape[0] == len(fitness)
        assert encodings.shape[0] == len(fitness_bin)

        self.encodings = torch.tensor(np.stack(encodings)).float().to(device)
        self.ids = ids.values
        self.fitness = torch.tensor(fitness.values).float().to(device)
        self.fitness_bin = torch.tensor(fitness_bin.values).float().to(device)

    def __len__(self):
        return self.encodings.shape[0]

    def __getitem__(self, index) -> Tuple[torch.tensor, float, str]:
        return (
            self.encodings[index, :],
            self.ids[index],
            self.fitness[index],
            self.fitness_bin[index],
        )
