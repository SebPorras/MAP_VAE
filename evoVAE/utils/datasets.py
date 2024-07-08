import numpy as np
from typing import Tuple
import torch
from torch.utils.data import Dataset


class MSA_Dataset(Dataset):

    def __init__(
        self, encodings: np.ndarray, weights: np.ndarray, ids: np.ndarray
    ) -> None:

        assert encodings.shape[0] == len(weights)
        assert encodings.shape[0] == len(ids)

        self.encodings = torch.tensor(np.stack(encodings))
        self.weights = torch.tensor(weights.values)
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
    ) -> None:

        assert encodings.shape[0] == len(ids)
        assert encodings.shape[0] == len(fitness)
        assert encodings.shape[0] == len(fitness_bin)

        self.encodings = torch.tensor(np.stack(encodings))
        self.ids = ids.values
        self.fitness = torch.tensor(fitness.values)
        self.fitness_bin = torch.tensor(fitness_bin.values)

    def __len__(self):
        return self.encodings.shape[0]

    def __getitem__(self, index) -> Tuple[torch.tensor, float, str]:
        return (
            self.encodings[index, :],
            self.ids[index],
            self.fitness[index],
            self.fitness_bin[index],
        )
