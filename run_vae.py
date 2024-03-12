# %%
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from evoVAE.trainer.checkerboard_train import train
from evoVAE.decoders.minimal_decoder import MinDecoder
from evoVAE.encoders.minimal_encoder import MinEncoder
from evoVAE.models.standardVAE import StandardVAE


# %%
training_data = datasets.FashionMNIST(
    root="data", train=True, download=False, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=False, transform=ToTensor()
)

# %%
train_loader = torch.utils.data.DataLoader(training_data, batch_size=128)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)


# %%
# standard input size for MINST data
inputDim = 28 * 28

bottleNeckDim = 512
latentDim = 2

encoder = MinEncoder(inputDim=inputDim, bottleNeckDim=bottleNeckDim)
decoder = MinDecoder(
    inputDim=bottleNeckDim, bottleNeckDim=bottleNeckDim, outputDim=inputDim
)

model = StandardVAE(
    inputDims=inputDim,
    bottleNeckDim=bottleNeckDim,
    latentDim=latentDim,
    encoder=encoder,
    decoder=decoder,
)


# %%
trained_model = train(model, train_loader, epochs=30, print_freq=10)
torch.save(trained_model.state_dict(), 'model_weights.pth')
