from __future__ import annotations
from .types_ import *
import torch
from torch import nn
from abc import abstractmethod


class BaseVAE(nn.Module):
    """Standard methods for the VAE class"""

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def configure_optimiser(self, *inputs: Any, **kwargs):
        pass
