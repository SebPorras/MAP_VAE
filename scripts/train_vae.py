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
from datetime import datetime

CONFIG_FILE = 1
ARRAY_ID = 2
ALIGN_FILE = -2
MULTIPLE_REPS = 3
HAS_REPLICATES = 0
SEQ_LEN = 0
BATCH_ZERO = 0
SEQ_ZERO = 0

# %% [markdown]
# #### Config

wandb.login()

start = time.time()

with open(sys.argv[CONFIG_FILE], "r") as stream:
    settings = yaml.safe_load(stream)

# slurm array task id
if len(sys.argv) == MULTIPLE_REPS:
    replicate = sys.argv[ARRAY_ID]
    unique_id = settings["info"] + "_r" + replicate + "/"
    settings["replicate"] = int(replicate)
else:
    unique_id = settings["info"] + "/"

settings["info"] = unique_id

# create output directory for data
if not os.path.exists(unique_id):
    os.mkdir(unique_id)

# %%
wandb.init(
    project=settings["project"],
    # hyperparameters
    config=settings,
    name=unique_id,
)


# %%
config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# #### Data loading and preprocessing

# %%

# Read in the datasets and create train and validation sets
ancestors_extants_aln = pd.read_pickle(config.alignment)

if config.replicate != HAS_REPLICATES:
    replicate_data = pd.read_csv(config.replicate_csv)

    # subset based on random sample
    indices = replicate_data["rep_" + str(config.replicate)]
    ancestors_extants_aln = ancestors_extants_aln.loc[indices]

# add weights to the sequences
numpy_aln, _, _ = st.convert_msa_numpy_array(ancestors_extants_aln)
weights = st.reweight_by_seq_similarity(numpy_aln, config.seq_theta)
ancestors_extants_aln["weights"] = weights
# one-hot encode
one_hot = ancestors_extants_aln["sequence"].apply(st.seq_to_one_hot)
ancestors_extants_aln["encoding"] = one_hot

train, val = train_test_split(ancestors_extants_aln, test_size=config.test_split)
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

# save config for the run 
yaml_str = yaml.dump(settings, default_flow_style=False)
with open(unique_id + "log.txt", "w") as file:
    file.write("run_id:", unique_id)
    file.write("time:", datetime.now())
    file.write("###CONFIG###")
    file.write(yaml_str)
    file.write(print(model))


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

extant_aln = pd.read_pickle(config.extant_aln)
# add weights to the sequences
numpy_aln, _, _ = st.convert_msa_numpy_array(extant_aln)
weights = st.reweight_by_seq_similarity(numpy_aln, config.seq_theta)
extant_aln["weights"] = weights
# one-hot encode
one_hot = extant_aln["sequence"].apply(st.seq_to_one_hot)
extant_aln["encoding"] = one_hot


pearson = calc_reconstruction_accuracy(
    trained_model, extant_aln, unique_id, config.latent_samples, config.num_processes
)

final_metrics = pd.read_csv(unique_id + "zero_shot_final_metrics.csv")
final_metrics["pearson"] = [pearson]
final_metrics.to_csv(unique_id + "zero_shot_final_metrics.csv")

print(f"elapsed minutes: {(time.time() - start) / 60}")

trained_model.cpu()
# remove '/' at end and './' at start of unique_id and save
torch.save(trained_model.state_dict(), unique_id + f"{unique_id[2:-1]}_model_state.pt")

wandb.finish()
# %%
