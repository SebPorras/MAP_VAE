# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: embed
#     language: python
#     name: python3
# ---

import MAP_VAE.utils.seq_tools as st
import MAP_VAE.utils.metrics as mt
from MAP_VAE.models.seqVAE import SeqVAE
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
import yaml
from MAP_VAE.loss.standard_loss import (
    KL_divergence,
    sequence_likelihood,
    elbo_importance_sampling,
)


# This notebook can be used to test new features for a model without having to use the WandB service

# +
with open("../data/dummy_config.yaml", "r") as stream:
    settings = yaml.safe_load(stream)
seq_len = 770  # A4 Human length
input_dims = seq_len * settings["AA_count"]


model = SeqVAE(
    dim_latent_vars=settings["latent_dims"],
    dim_msa_vars=input_dims,
    num_hidden_units=settings["hidden_dims"],
    settings=settings,
    num_aa_type=settings["AA_count"],
)

# device = "cpu"
# model.load_state_dict(torch.load("a4_extants_r1_model_state.pt", map_location=device))
# model
# -

metadata = pd.read_csv("../data/DMS_substitutions.csv")
dms_data = pd.read_csv("A4_HUMAN_Seuma_2022.csv")
one_hot = dms_data["mutated_sequence"].apply(st.seq_to_one_hot)
dms_data["encoding"] = one_hot

#

# +
from MAP_VAE.utils.datasets import DMS_Dataset, MSA_Dataset

device = torch.device("cpu")

aln = st.read_aln_file(
    "/Users/sebs_mac/uni_OneDrive/honours/data/a4_human/alns/a4_extants_no_dupes.fasta"
)
one_hot = aln["sequence"].apply(st.seq_to_one_hot)
aln["encoding"] = one_hot


numpy_aln, _, _ = st.convert_msa_numpy_array(aln)
weights = st.position_based_seq_weighting(numpy_aln, n_processes=10)
aln["weights"] = weights

aln = aln.loc[range(10)]
msas = MSA_Dataset(aln["encoding"], aln["weights"], aln["id"], device)
train_loader = torch.utils.data.DataLoader(msas, batch_size=1, shuffle=True)


# +
num_samples = 10
ebls = []
with torch.no_grad():

    for (
        x,
        _,
        id,
    ) in train_loader:

        elbo = model.compute_elbo_with_multiple_samples(x, 5000)
        ebls.append(elbo)


np.mean(ebls)
