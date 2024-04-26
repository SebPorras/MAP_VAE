# %%
from evoVAE.utils.datasets import MSA_Dataset
import evoVAE.utils.seq_tools as st
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seq_trainer import seq_train
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
        "alignment": "../data/GFP_AEQVI_full_04-29-2022_b08_encoded_weighted_ancestors_extants_no_syn.pkl",
        "seq_theta": 0.2,  # reweighting
        "AA_count": 21,  # standard AA + gap
        "test_split": 0.2,
        "max_mutation": 4,  # how many mutations the model will test up to
        # ADAM
        "learning_rate": 1e-2,  # ADAM
        "weight_decay": 1e-4,  # ADAM
        # Hidden units
        "momentum": None,
        "dropout": None,
        # Training loop
        "epochs": 60,
        "batch_size": 128,
        "max_norm": 5.0,  # gradient clipping
        # Model info - default settings
        "architecture": "SeqVAE",
        "latent_dims": 10,
        "hidden_dims": [128, 64, 32],
        # DMS data
        "dms_file": "../data/GFP_AEQVI_Sarkisyan_2016_dms_encoded.pkl",
        "dms_metadata": "../data/DMS_substitutions.csv",
        "dms_id": "GFP_AEQVI_Sarkisyan_2016",
    },
)


config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# #### Data loading and preprocessing

# %%

# Read in the datasets and create train and validation sets
# Assume that encodings and weights have been calculated.
# outgroup ancestors have already been removed.
ancestors_aln = pd.read_pickle(config.alignment)
train, val = train_test_split(ancestors_aln, test_size=config.test_split)

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

# Load and process the DMS data used for fitness prediction
dms_data = pd.read_pickle(config.dms_file)
metadata = pd.read_csv(config.dms_metadata)
# grab metadata for current experiment
metadata = metadata[metadata["DMS_id"] == config.dms_id]

# %% [markdown]
# #### Create the model

# %%
# get the sequence length from first sequence
SEQ_LEN = 0
BATCH_ZERO = 0
SEQ_ZERO = 0
seq_len = train_dataset[BATCH_ZERO][SEQ_ZERO].shape[SEQ_LEN]
input_dims = seq_len * config.AA_count

# instantiate the model
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
    dms_data=dms_data,
    metadata=metadata,
    device=device,
    config=config,
)

# %%
wandb.finish()
