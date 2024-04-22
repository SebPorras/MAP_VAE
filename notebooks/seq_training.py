# %%
from evoVAE.utils.datasets import MSA_Dataset
import evoVAE.utils.seq_tools as st
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seqVAE_train import seq_train
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import wandb
from pathlib import Path
import os

# %% [markdown]
# Contains a very rough overview of how training could be done. Refer to train_seqVAE.py for a more complete workflow

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
        "momentum": 0.1,
        "dropout": 0.5,
        # Training loop
        "epochs": 100,
        "batch_size": 128,
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

DATA_PATH = Path(
    "/Users/sebs_mac/OneDrive - The University of Queensland/honours/data/phoQ/uniref90_search/nr65_filtering/odseq_tree/independent_runs/ancestors"
)

# Gather all the ancestor sequences into a single dataframe
trees = []
for file in os.listdir(DATA_PATH):
    if file == "ancestor_trees":
        continue
    run = st.read_aln_file(str(DATA_PATH) + "/" + file)
    run["tree"] = file.split("_")[1]
    trees.append(run)

ancestors = pd.concat(trees)
anc_encodings, anc_weights = st.encode_and_weight_seqs(ancestors["sequence"], theta=0.2)
ancestors["weights"] = anc_weights
# ancestors.to_pickle("phoQ_ancestors_weights.pkl")


# %%
# Next, drop N0 and N238 as they come from outgroups
print(ancestors.shape)
flt_ancestors = ancestors.loc[(ancestors["id"] != "N0") & (ancestors["id"] != "N238")]
print(flt_ancestors.shape)

# Then remove non-unique sequences
flt_unique_ancestors = flt_ancestors.drop_duplicates(subset="sequence")
flt_unique_ancestors


# %%
flt_unique_ancestors = st.read_aln_file("../data/alignments/tiny.aln")
anc_encodings, anc_weights = st.encode_and_weight_seqs(
    flt_unique_ancestors["sequence"], theta=config.seq_theta
)
flt_unique_ancestors["weights"] = anc_weights
flt_unique_ancestors["encodings"] = anc_encodings


train, val = train_test_split(flt_unique_ancestors, test_size=0.1)

# create one-hot encodings and calculate reweightings

# TRAINING
train_dataset = MSA_Dataset(train["encodings"], train["weights"], train["id"])

# VALIDATION
val_dataset = MSA_Dataset(val["encodings"], val["weights"], val["id"])

# DATA LOADERS #
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

print(len(train_loader), len(val_loader))
next(iter(train_loader))[0].shape, next(iter(train_loader))[1].shape, next(
    iter(train_loader)
)[2]

# %% [markdown]
# #### Create the model

# %%
# get the sequence length
SEQ_LEN = 0
BATCH_ZERO = 0
SEQ_ZERO = 0
seq_len = train_dataset[BATCH_ZERO][SEQ_ZERO].shape[SEQ_LEN]
input_dims = seq_len * config.AA_count

# use preset structure for hidden dimensions
model = SeqVAE(
    input_dims=input_dims,
    latent_dims=config.latent_dims,
    hidden_dims=config.hidden_dims,
    config=config,
)
model

# %%

for i in train_loader:
    encoding, weight, name = i

    print(encoding.shape)

    # encoding = encoding.float()
    # output = model.forward(encoding)
    # print(encoding.shape, output[0].shape)
    # loss, kl, likelihood = model.loss_function(output, encoding)
    # print(loss, kl, likelihood)


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
a = torch.rand(size=(2, 2))
b = torch.rand(size=(2, 2))

c = a - b
print(c)
print(c.sum(-1))
print(c.sum(dim=tuple(range(1, c.ndim))))

# %%
wandb.finish()
