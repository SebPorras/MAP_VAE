import torch
from torch import nn


class MinEncoder(nn.Module):

    def __init__(self, inputDim: int, bottleNeckDim: int) -> None:
        super(MinEncoder, self).__init__()

        self.inputDim = inputDim

        # hyper parameters
        self.bottleNeckDim = bottleNeckDim

        # encoder layers
        self.linearReluStack = nn.Sequential(
            nn.Linear(inputDim, bottleNeckDim),
            nn.ReLU(),
            nn.Linear(bottleNeckDim, bottleNeckDim),
            nn.ReLU(),
        )

    def forward(self, rawInput: torch.Tensor) -> torch.Tensor:
        return self.linearReluStack(rawInput)
