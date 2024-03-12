import torch
from torch import nn

class MinDecoder(nn.Module):

    def __init__(self, inputDim: int, bottleNeckDim: int, outputDim: int) -> None:
        super(MinDecoder, self).__init__()

        self.inputDim = inputDim
        self.bottleNeckDim = bottleNeckDim
        self.outputDim = outputDim

        # decoder layers
        self.linearReluStack = nn.Sequential(
            nn.Linear(inputDim, bottleNeckDim),
            nn.ReLU(),
            nn.Linear(bottleNeckDim, outputDim),
            nn.Softmax(),
        )

    def forward(self, latentInput: torch.Tensor) -> torch.Tensor:
        xHat = self.linearReluStack(latentInput)
        return xHat
