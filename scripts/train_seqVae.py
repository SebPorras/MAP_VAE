# %%
from evoVAE.utils.datasets import MSA_Dataset
import evoVAE.utils.seq_tools as st
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seqVAE_train import seq_train
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import wandb

# %% [markdown]
# #### Config

# %%
wandb.init(
    project="SeqVAE_training",
    # hyperparameters
    config={
        # Dataset info
        "dataset": "PhoQ",
        "seq_theta": 0.2,  # reweighting
        "AA_count": 21,  # standard AA + gap
        # ADAM
        "learning_rate": 1e-5,  # ADAM
        "weight_decay": 0.01,  # ADAM
        # Hidden units
        "momentum": 0.9,
        "dropout": 0.5,
        # Training loop
        "epochs": 1,
        "batch_size": 2,
        "max_norm": 1.0,  # gradient clipping
        # Model info
        "architecture": "SeqVAE",
        "latent_dims": 2,
        "hidden_dims": [32, 16],
    },
)

config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# #### Data loading and preprocessing

# %%

# Read in the datasets and create train and validation sets
alns: pd.DataFrame = st.read_aln_file("../data/alignments/tiny.aln")
train, val = train_test_split(alns, test_size=0.2)

# create one-hot encodings and calculate reweightings

# TRAINING
train_encodings, train_weights = st.encode_and_weight_seqs(
    train["sequence"], theta=config.seq_theta
)
train_ids = train["id"].values  # just the seq identifiers
train_dataset = MSA_Dataset(train_encodings, train_weights, train_ids)

# VALIDATION
val_encodings, val_weights = st.encode_and_weight_seqs(
    val["sequence"], theta=config.seq_theta
)
val_ids = val["id"].values
val_dataset = MSA_Dataset(val_encodings, val_weights, val_ids)


# DATA LOADERS #
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=False
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False
)
# next(iter(train_loader))[0].shape,next(iter(train_loader))[1].shape, next(iter(train_loader))[2]

# %%
# encoding, weights, id = train_dataset[0]
# print(encoding.shape, weights, id)

# translation = st.one_hot_to_seq(encoding)
# print(translation)

# %% [markdown]
# #### Create the model

# %%
# get the sequence length
seq_len = train_dataset[0][0].shape[0]
input_dims = seq_len * config.AA_count

# use preset structure for hidden dimensions
model = SeqVAE(
    input_dims=input_dims,
    latent_dims=config.latent_dims,
    hidden_dims=config.hidden_dims,
    config=config,
)
# model

# %% [markdown]
# #### Training Loop

# %%
trained_model = seq_train(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    config=config,
)

# %%
wandb.finish()
