# %%
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seq_trainer import seq_train, calc_reconstruction_accuracy
from sklearn.model_selection import train_test_split
import pandas as pd
import evoVAE.utils.seq_tools as st
import torch
import wandb
import sys, yaml, time
import os

# %% [markdown]
# #### Config
CONFIG_FILE = 1
ARRAY_ID = 2
ALIGN_FILE = -2
start = time.time()

with open(sys.argv[CONFIG_FILE], "r") as stream:
    settings = yaml.safe_load(stream)

# slurm array task id
if len(sys.argv) == 3:
    replicate = sys.argv[ARRAY_ID]

unique_id = settings["info"] + "_r" + replicate + "/"
settings["info"] = unique_id

if not os.path.exists(unique_id):
    os.mkdir(unique_id)


settings["replicate"] = int(replicate)

# %%
wandb.init(
    project=settings["project"],
    # hyperparameters
    config=settings,
)


# %%
config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# #### Data loading and preprocessing

# %%

# Read in the datasets and create train and validation sets
# Assume that encodings and weights have been calculated.
ancestors_aln = pd.read_pickle(config.alignment)

if config.replicate_csv != 0:
    replicate_data = pd.read_csv(config.replicate_csv)

    indices = replicate_data.loc[
        replicate_data["replicate"] == config.replicate, "indices"
    ].values[0]

    indices = [int(x.strip()) for x in indices[1:-1].split(",")]

    # subset based on random sample
    ancestors_aln = ancestors_aln.loc[indices]

# one-hot encode and add weights to teh sequences
numpy_aln, _, _ = st.convert_msa_numpy_array(ancestors_aln)
weights = st.reweight_by_seq_similarity(numpy_aln, config.seq_theta)
ancestors_aln["weights"] = weights
one_hot = ancestors_aln["sequence"].apply(st.seq_to_one_hot)
ancestors_aln["encoding"] = one_hot

train, val = train_test_split(ancestors_aln, test_size=config.test_split)
print(f"Train shape: {train.shape}")
print(f"Validation shape: {val.shape}")

# TRAINING
train_dataset = MSA_Dataset(train["encoding"], train["weights"], train["id"])

# VALIDATION
val_dataset = MSA_Dataset(val["encoding"], val["weights"], val["id"])

# DATA LOADERS #
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=True
)

# Load and subset the DMS data used for fitness prediction
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
print(f"Seq length: {seq_len}")

# instantiate the model
model = SeqVAE(
    input_dims=input_dims,
    latent_dims=config.latent_dims,
    hidden_dims=config.hidden_dims,
    config=config,
)
# model
print(model)

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
    unique_id=unique_id,
)

# extant_aln = pd.read_pickle(config.extant_aln)
# calc_reconstruction_accuracy(
#    trained_model, extant_aln, unique_id, config.latent_samples, config.num_processes
# )

print(f"elapsed minutes: {(time.time() - start) / 60}")
wandb.finish()

trained_model.cpu()
torch.save(trained_model.state_dict(), unique_id + f"{unique_id[2:]}_model_state.pt")

# %%
