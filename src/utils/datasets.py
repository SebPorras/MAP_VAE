"""
datasets.py
Custom PyTorch Dataset classes for handling data.
"""

import numpy as np
from typing import Tuple
import torch
from torch.utils.data import Dataset


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
    """
    Holds data for a deep mutational scanning dataset.

    Attributes:
        encodings (torch.Tensor): Tensor containing the encoded data.
        ids (np.ndarray): Array containing the identifiers for each data point.
        fitness (torch.Tensor): Tensor containing the fitness values.
        fitness_bin (torch.Tensor): Tensor containing the binned fitness values.
        device (torch.device): The device on which tensors are stored.

        Args:
            encodings (np.ndarray): Array of encoded data.
            ids (np.ndarray): Array of identifiers.
            fitness (np.ndarray): Array of fitness values.
            fitness_bin (np.ndarray): Array of binned fitness values.
            device (torch.device): The device to store the tensors on.
    """

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
