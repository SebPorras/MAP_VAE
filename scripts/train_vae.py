# %%
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAEv2 import SeqVAE
from evoVAE.models.tanh_vae import tanhVAE
from evoVAE.trainer.seq_trainer import seq_train, calc_reconstruction_accuracy
from sklearn.model_selection import train_test_split
import pandas as pd
import evoVAE.utils.seq_tools as st
import torch
import wandb
import sys, yaml, time
import os
from datetime import datetime
import matplotlib.pyplot as plt

CONFIG_FILE = 1
ARRAY_ID = 2
ALIGN_FILE = -2
MULTIPLE_REPS = 3
SEQ_LEN = 0
BATCH_ZERO = 0
SEQ_ZERO = 0

# %% [markdown]
# #### Config

# os.environ["WANDB_MODE"] = "offline"
# wandb.login()

start = time.time()

with open(sys.argv[CONFIG_FILE], "r") as stream:
    settings = yaml.safe_load(stream)

# slurm array task id
if len(sys.argv) == MULTIPLE_REPS:
    replicate = sys.argv[ARRAY_ID]
    unique_id_path = settings["info"] + "_r" + replicate + "/"
    settings["replicate"] = int(replicate)
else:
    unique_id_path = settings["info"] + "/"

# remove '/' at end and './' at start of unique_id
unique_id = unique_id_path[2:-1]
settings["info"] = unique_id_path

# create output directory for data
if not os.path.exists(unique_id_path):
    os.mkdir(unique_id_path)

# %%
# wandb.init(
#     project=settings["project"],
#     # hyperparameters
#     config=settings,
#     name=unique_id,
# )


# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #### Data loading and preprocessing

# Read in the datasets and create train and validation sets
ancestors_extants_aln = pd.read_pickle(settings["alignment"])

if settings["replicate_csv"] is not None:
    replicate_data = pd.read_csv(settings["replicate_csv"])
    # subset based on random sample
    indices = replicate_data["rep_" + str(settings["replicate"])]
    ancestors_extants_aln = ancestors_extants_aln.loc[indices]

# add weights to the sequences
numpy_aln, _, _ = st.convert_msa_numpy_array(ancestors_extants_aln)
weights = st.position_based_seq_weighting(numpy_aln, n_processes=8)
ancestors_extants_aln["weights"] = weights
# one-hot encode
one_hot = ancestors_extants_aln["sequence"].apply(st.seq_to_one_hot)
ancestors_extants_aln["encoding"] = one_hot

train, val = train_test_split(ancestors_extants_aln, test_size=settings["test_split"])


log = ""
log += f"Train shape: {train.shape}\n"
log += f"Validation shape: {val.shape}\n"

# TRAINING
train_dataset = MSA_Dataset(train["encoding"], train["weights"], train["id"])

# VALIDATION
val_dataset = MSA_Dataset(val["encoding"], val["weights"], val["id"])

# DATA LOADERS #
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=settings["batch_size"], shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=settings["batch_size"], shuffle=True
)

# Load and subset the DMS data used for fitness prediction
dms_data = pd.read_pickle(settings["dms_file"])
metadata = pd.read_csv(settings["dms_metadata"])
# grab metadata for current experiment
metadata = metadata[metadata["DMS_id"] == settings["dms_id"]]

# %% [markdown]
# #### Create the model

# %%
# get the sequence length from first sequence

seq_len = train_dataset[BATCH_ZERO][SEQ_ZERO].shape[SEQ_LEN]
input_dims = seq_len * settings["AA_count"]
log += f"Seq length: {seq_len}\n"

# instantiate the model
"""
model = tanhVAE(
     dim_latent_vars=2,
     dim_msa_vars=input_dims,
     num_hidden_units=[150, 150],
     num_aa_type=21,
)

"""
model = SeqVAE(
    dim_latent_vars=settings["latent_dims"],
    dim_msa_vars=input_dims,
    num_hidden_units=settings["hidden_dims"],
    settings=settings,
    num_aa_type=settings["AA_count"],
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
    config=settings,
    unique_id=unique_id_path,
)

# plot the loss for visualtion of learning
losses = pd.read_csv(unique_id_path + "loss.csv")

plt.figure()
plt.plot(losses["epoch"], losses["elbo"], label="train", marker="o", color="b")
plt.plot(losses["epoch"], losses["val_elbo"], label="validation", marker="x", color="r")
plt.xlabel("epoch")
plt.ylabel("ELBO")
plt.legend()
plt.title(unique_id)  # remove file / and ./
plt.savefig(unique_id_path + "loss" + ".png")


# save config for the run
yaml_str = yaml.dump(settings, default_flow_style=False)
with open(unique_id_path + "log.txt", "w") as file:
    file.write(f"run_id: {unique_id_path}\n")
    file.write(f"time: {datetime.now()}\n")
    file.write("###CONFIG###\n")
    file.write(f"{yaml_str}\n")
    file.write(log)
    file.write(f"{str(model)}\n")
    file.write(f"elapsed minutes: {(time.time() - start) / 60}\n")


torch.save(
    trained_model.state_dict(),
    unique_id_path + f"{unique_id}_model_state.pt",
)

# wandb.finish()
# %%
