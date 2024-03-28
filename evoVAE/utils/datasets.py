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
    """Hold data for a deep mutational scanning dataset"""
    
    def __init__(
        self, encodings: np.ndarray, weights: np.ndarray, ids: np.ndarray, 
        fitness: np.ndarray) -> None: 
        
        assert encodings.shape[0] == len(ids)
        assert encodings.shape[0] == len(fitness)
        
        self.encodings = torch.tensor(np.stack(encodings))
        self.ids = ids
        self.fitness = torch.tensor(fitness.values)
        
    def __len__(self):
        return self.encodings.shape[0]

    def __getitem__(self, index) -> Tuple[torch.tensor, float, str]:
        return self.encodings[index, :],  self.fitness[index], self.ids[index],
        
    