# %%
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seq_trainer import seq_train, calc_reconstruction_accuracy
from sklearn.model_selection import train_test_split
import pandas as pd
import evoVAE.utils.seq_tools as st
import torch
import sys, yaml, time
import os

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


extant_aln = pd.read_pickle(settings["extant_aln"])

# add weights to the sequences
numpy_aln, _, _ = st.convert_msa_numpy_array(extant_aln)
weights = st.reweight_by_seq_similarity(numpy_aln, settings["seq_theta"])
extant_aln["weights"] = weights

# one-hot encode
one_hot = extant_aln["sequence"].apply(st.seq_to_one_hot)
extant_aln["encoding"] = one_hot


seq_len = numpy_aln.shape[1]
input_dims = seq_len * settings["AA_count"]
print(f"Seq length: {seq_len}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# instantiate the model
model = SeqVAE(
    input_dims=input_dims,
    latent_dims=settings["latent_dims"],
    hidden_dims=settings["hidden_dims"],
    config=settings,
)

model.load_state_dict(torch.load(unique_id + f"{unique_id[2:-1]}_model_state.pt", map_location=device))

# %% [markdown]
# #### Training Loop

pearson = calc_reconstruction_accuracy(
    model, extant_aln, unique_id, settings[latent_samples, settings[num_processes
)

final_metrics = pd.read_csv(unique_id + "zero_shot_all_variants_final_metrics.csv")
final_metrics["pearson"] = [pearson]
final_metrics.to_csv(unique_id + "zero_shot_all_variants_final_metrics.csv")

print(f"elapsed minutes: {(time.time() - start) / 60}")

# remove '/' at end and './' at start of unique_id and save

wandb.finish()
# %%
