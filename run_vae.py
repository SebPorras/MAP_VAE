# %%
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from evoVAE.trainer.convo_train import train
from evoVAE.models.convoVAE import ConvoVAE


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
model = ConvoVAE(in_channels=1, latentDims=10)

# %%
trained_model = train(model, train_loader, epochs=40, print_freq=10)
torch.save(trained_model.state_dict(), "model_weights.pth")
