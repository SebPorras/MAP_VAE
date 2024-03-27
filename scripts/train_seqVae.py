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
        "test_split": 0.2,
        # ADAM
        "learning_rate": 1e-5,  # ADAM
        "weight_decay": 0.01,  # ADAM
        # Hidden units
        "momentum": 0.1,
        "dropout": 0.5,
        # Training loop
        "epochs": 500,
        "batch_size": 128,
        "max_norm": 1.0,  # gradient clipping
        # Model info
        "architecture": "SeqVAE",
        "latent_dims": 4,
        "hidden_dims": [128, 64],
    },
)

config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# #### Data loading and preprocessing

# %%

# Read in the datasets and create train and validation sets
# ancestors_df = pd.read_pickle("../data/alignments/phoQ_ancestors.pkl")
# anc_encodings, anc_weights = st.encode_and_weight_seqs(
#    ancestors_df["sequence"], theta=config.seq_theta
# )
# ancestors_df["weights"] = anc_weights
# ancestors_df["encodings"] = anc_encodings
# ancestors_df.to_pickle("phoQ_ancestors_weights_encodings.pkl")

ancestors_df = pd.read_pickle("phoQ_ancestors_weights_encodings.pkl")

# Next, drop N0 and N238 as they come from outgroups
flt_ancestors = ancestors_df.loc[
    (ancestors_df["id"] != "N0") & (ancestors_df["id"] != "N238")
]

# Then remove non-unique sequences
flt_unique_ancestors = flt_ancestors.drop_duplicates(subset="sequence")

train, val = train_test_split(flt_unique_ancestors, test_size=config.test_split)

# TRAINING
train_dataset = MSA_Dataset(train["encodings"], train["weights"], train["id"])

# VALIDATION
val_dataset = MSA_Dataset(val["encodings"], val["weights"], val["id"])

# DATA LOADERS #
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=False
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False
)

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
