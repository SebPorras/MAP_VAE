import torch
from trainer.checkerboard_train import *
from decoders.minimal_decoder import MinDecoder
from encoders.minimal_encoder import MinEncoder
from models.standardVAE import StandardVAE


# ! which python3

def main():
    """
    inputDim = 2
    bottleNeckDim = 1024
    latentDim = 2

    encoder = MinEncoder(inputDim=inputDim, bottleNeckDim=bottleNeckDim)
    decoder = MinDecoder(inputDim=inputDim, bottleNeckDim=bottleNeckDim)

    model = StandardVAE(
        bottleNeckDim=bottleNeckDim,
        latentDim=latentDim,
        encoder=encoder,
        decoder=decoder,
    )
    """

    # model = ck.train("checkerboard", model=model, epochs=60000, print_freq=1000)
    print(sample2d("checkerboard"))


