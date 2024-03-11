import evoVAE.loss.checkerboard_train as ck

from evoVAE.decoders.minimal_decoder import MinDecoder
from evoVAE.encoders.minimal_encoder import MinEncoder
from evoVAE.models.standardVAE import StandardVAE


def main():

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

    ck.train("checkerboard", model=model, epochs=60000, print_freq=1000)


if __name__ == "__main__":
    main()
