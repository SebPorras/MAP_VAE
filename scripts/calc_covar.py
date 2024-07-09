# %%
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seq_trainer import calc_reconstruction_accuracy
import pandas as pd
import evoVAE.utils.seq_tools as st
import torch
import sys, yaml, time
import os

CONFIG_FILE = 1
ARRAY_ID = 2
ALIGN_FILE = -2
MULTIPLE_REPS = 3
PROCESSES = 4
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
    unique_id_path = settings["info"] + "_r" + replicate + "/"
    settings["replicate"] = int(replicate)
else:
    unique_id_path = settings["info"] + "/"

unique_id = unique_id_path[2:-1]
settings["info"] = unique_id_path

# create output directory for data
if not os.path.exists(unique_id_path):
    os.mkdir(unique_id_path)

# Read in the datasets and create train and validation sets
if settings["extant_aln"].split(".")[-1] in ["fasta", "aln"]:
    extant_aln = st.read_aln_file(settings["extant_aln"])
else:
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
    dim_latent_vars=settings["latent_dims"],
    dim_msa_vars=input_dims,
    num_hidden_units=settings["hidden_dims"],
    settings=settings,
    num_aa_type=settings["AA_count"],
)

model.load_state_dict(
    torch.load(unique_id_path + f"{unique_id}_model_state.pt", map_location=device)
)
model.to(device)

# %% [markdown]
# #### Training Loop

pearson = calc_reconstruction_accuracy(
    model,
    extant_aln,
    unique_id_path,
    device,
    num_samples=50,
    num_processes=int(os.getenv("SLURM_CPUS_PER_TASK")),
)

final_metrics = pd.read_csv(unique_id_path + "zero_shot_all_variants_final_metrics.csv")
final_metrics["pearson"] = [pearson]
final_metrics.to_csv(
    unique_id_path + "zero_shot_all_variants_final_metrics.csv", index=False
)

print(f"elapsed minutes: {(time.time() - start) / 60}")
