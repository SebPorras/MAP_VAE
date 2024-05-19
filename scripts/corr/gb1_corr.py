# gb1_corr.py

import evoVAE.utils.seq_tools as st
import pandas as pd
from evoVAE.utils.seq_tools import GAPPY_ALPHABET_LEN
import numpy as np
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAETest import SeqVAETest
import torch


def pair_wise_covariances(msa):

    SEQ_COUNT = 0
    COLS = 1

    pairs = []
    for i in range(st.GAPPY_ALPHABET_LEN):
        pairs.extend([(i, j) for j in range(i + 1, st.GAPPY_ALPHABET_LEN)])

    # the number of unique ways we can compare columns in the MSA
    column_combinations = msa.shape[COLS] * (msa.shape[COLS] - 1) // 2
    # number of different residue combinations we can have
    aa_combinations = GAPPY_ALPHABET_LEN**2

    num_seqs = msa.shape[SEQ_COUNT]
    num_columns = msa.shape[COLS]

    # each column has aa_combinations many ways to combine residues
    # this is an upper triangular matrix but we will store it in a linear format.
    covariances = np.zeros(column_combinations * aa_combinations)

    # keep track of which column combination we're up to
    col_combination_count = 0
    for i in range(num_columns - 1):
        for j in range(i + 1, num_columns):
            col_i = msa[:, i]
            col_j = msa[:, j]

            for a, b in pairs:

                # find how many sequences have residues a and b
                col_i_res = np.where(col_i == a)[0]
                col_j_res = np.where(col_j == b)[0]

                # find how many sequences have this combination
                intersect = np.intersect1d(
                    col_i_res, col_j_res, assume_unique=True
                ).shape[SEQ_COUNT]
                # make a frequency based on number of sequences
                freq_Ai_Bj = intersect / num_seqs

                # just count how many sequences have these residues
                freq_Ai = col_i_res.shape[0] / num_seqs
                freq_Bj = col_j_res.shape[0] / num_seqs

                # get correct position: (which column combination we're at) + (which residue combination we're at)
                covar_index = (
                    col_combination_count * aa_combinations
                    + a * st.GAPPY_ALPHABET_LEN
                    + b
                )

                # useful in case you want to find a specific cov score based on column and residue indices in the upper tri matrix
                # col_combination_count = (num_cols*(num_cols-1)/2) - (num_cols-col_1_idx)*((num_cols-col_1_idx)-1)/2 + col_2_idx - col_1_idx - 1
                # covar_index = int(col_combination_count * aa_combinations + a_idx * st.GAPPY_ALPHABET_LEN + b_idx)

                # (joint occurances of residues a & b at thi) - (occurence of A at col i * occurence of B at col j)
                covariances[covar_index] = freq_Ai_Bj - (freq_Ai * freq_Bj)

            # keep track of how many column combinations we've seen
            col_combination_count += 1

    return covariances


aln_file = "../data/pair_test.aln"
orig_aln = pd.read_pickle(aln_file)
#aln = orig_aln.sample(n=3000, random_state=42)


train_dataset = MSA_Dataset(aln["encoding"], aln["weights"], aln["id"])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)

SEQ_LEN = 0
BATCH_ZERO = 0
SEQ_ZERO = 0
seq_len = train_dataset[BATCH_ZERO][SEQ_ZERO].shape[SEQ_LEN]
input_dims = seq_len * 21


config = {
    # Dataset info
    "alignment": "tets",
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
    "epochs": 500,
    "batch_size": 128,
    "max_norm": 10,  # gradient clipping
    "patience": 3,
    # Model info - default settings
    "architecture": f"SeqVAE_0.25_ancestors_R",
    "latent_dims": 2,
    "hidden_dims": [256, 128, 64],
    # DMS data
    "dms_file": "../data/SPG1_STRSG_Wu_2016.pkl",
    "dms_metadata": "../data/DMS_substitutions.csv",
    "dms_id": "SPG1_STRSG_Wu_2016",
}


SEQ_LEN = 0
BATCH_ZERO = 0
SEQ_ZERO = 0
seq_len = train_dataset[BATCH_ZERO][SEQ_ZERO].shape[SEQ_LEN]
input_dims = seq_len * 21


model_weights = (
    "../data/gfp/model_weights/ancestors_extants_no_duplicates_gfp_model_state.pt"
)
model = SeqVAETest(input_dims, 2, hidden_dims=config["hidden_dims"], config=config)
model.load_state_dict(torch.load(model_weights))
model.eval()

names = []
z_vals = []

num_samples = 50
for encoding, weights, name in train_loader:

    encoding = encoding.float()
    encoding = torch.flatten(encoding, start_dim=1)
    # print(encoding.shape)

    encoding = encoding.expand(num_samples, encoding.shape[0], encoding.shape[1])
    # print(encoding.shape)
    # print(encoding.shape)
    z_mu, z_logvar = model.encode(encoding.float())
    z_samples = model.reparameterise(z_mu, z_logvar)
    # print(z_samples.shape)
    # print(z_samples[:, 0, :])
    mean_z = torch.mean(z_samples, dim=0)

    names.extend(name)
    z_vals.extend(mean_z.detach().numpy())


id_to_z = pd.DataFrame({"taxa": names, "z": z_vals})


recons = []
ids = []
# EVALUATE DIFFERENCES BETWEEN THE RECONSTRUCTIONS AND INPUT
for id, z in zip(id_to_z["taxa"], id_to_z["z"]):

    ids.append(id)
    # decode the Z sample and get it into a PPM shape
    x_hat = model.decode(torch.tensor(z))
    x_hat = x_hat.unsqueeze(-1)
    # print(x_hat.shape)

    x = aln[aln["id"] == id]["sequence"].values[0]
    x_one_hot = torch.tensor(st.seq_to_one_hot(x))
    # print(x_one_hot.shape)

    orig_shape = tuple(x_one_hot.shape)
    x_hat = x_hat.view(orig_shape)

    indices = x_hat.max(dim=-1).indices.tolist()
    recon = "".join([st.GAPPY_PROTEIN_ALPHABET[x] for x in indices])
    recons.append(recon)


recons_df = pd.DataFrame({"id": ids, "sequence": recons})
recon_msa, _, _ = st.convert_msa_numpy_array(recons_df)
predicted = pair_wise_covariances(recon_msa)

msa, _, _ = st.convert_msa_numpy_array(aln)
actual = pair_wise_covariances(msa)


df = pd.DataFrame({"actual": actual, "prediction": predicted})
df.to_csv("gb1_3000_corv.csv")
