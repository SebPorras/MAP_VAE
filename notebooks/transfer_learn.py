# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: embed
#     language: python
#     name: python3
# ---

# %%
import evoVAE.utils.visualisation as vs 
import evoVAE.utils.seq_tools as st
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.utils.datasets import MSA_Dataset
import pandas as pd
import torch
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %%
with open("../data/dummy_config.yaml", "r") as stream:
    settings = yaml.safe_load(stream)

# %%
dms_path = "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/"
variant_seqs = pd.read_csv(dms_path + "GFP_AEQVI_Sarkisyan_2016.csv")
variant_seqs.head()

# %%
variant_seqs[variant_seqs["mutant"] == "K3E"]

# %%

variant_seqs["encoding"] =  variant_seqs["mutated_sequence"].apply(st.seq_to_one_hot)
num_seqs = len(variant_seqs["mutated_sequence"])


device = torch.device("mps")
tree_dataset = MSA_Dataset(
    variant_seqs["encoding"],
    np.arange(len(variant_seqs["encoding"])),
    variant_seqs["mutant"],
    device=device,
)


# %%
tree_loader = torch.utils.data.DataLoader(
    tree_dataset, batch_size=num_seqs, shuffle=False
)
seq_len = tree_dataset[0][0].shape[0]
input_dims = seq_len * 21

# %%
model = SeqVAE(
    dim_latent_vars=settings["latent_dims"],
    dim_msa_vars=input_dims,
    num_hidden_units=settings["hidden_dims"],
    settings=settings,
    num_aa_type=settings["AA_count"],
)
model = model.to(device)

state_dict = "/Users/sebs_mac/gfp_ae_5_fold_wd_0.0_r1_wd_0.0_model_state.pt"
model.load_state_dict(torch.load(state_dict, map_location=device))

# latent = get_mean_z(model, tree_loader, device, n_samples)
latent = vs.get_mu(model, tree_loader)

# %%
latent.rename(columns={"id": "mutant"}, inplace=True)


# %%
merged = pd.merge(variant_seqs, latent, on="mutant")


# %%
merged.drop(columns=["encoding"], inplace=True)
merged

# %%
gfp_train, gfp_test = train_test_split(merged, test_size=0.2, random_state=42)

gfp_train.shape, gfp_test.shape

# %%
gfp_train

# %%
gfp_test

# %%
oh = np.stack(gfp_train["mutated_sequence"].apply(st.seq_to_one_hot))

data_tuples = oh.tolist()  # Convert to a list of lists of lists

df = pd.DataFrame(data_tuples, columns=[f'column_{i}' for i in range(238)])


df

# %%
con = pd.concat([gfp_train, df], join="outer")
con

# %%
from sklearn.linear_model import RidgeCV
ridge_model = RidgeCV(alphas=np.logspace(-6, 6, 1000), gcv_mode="auto")
X = np.array([np.array(x) for x in gfp_train["mu"]])
y = gfp_train["DMS_score"].values

# %%

# %%
y.values

# %%



ridge_model.fit(X, y)



X_test = np.array([np.array(x) for x in gfp_test["mu"]])
y_test = gfp_test["DMS_score"]
x_hat = ridge_model.predict(X_test)
print(np.corrcoef(x_hat, y_test))

plt.scatter(x_hat, y_test)
plt.show()

# %%
X = np.array([st.seq_to_one_hot(x) for x in  gfp_train["mutated_sequence"]])
y = gfp_train["DMS_score"].values

# %%
X.shape, y.shape

# %%
data = {f"col_{i}": list(X[:, i]) for i in range(238)}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# %%
ridge_model_oh = RidgeCV(alphas=np.logspace(-6, 6, 1000), gcv_mode="auto")
X = np.array([st.seq_to_one_hot(x).flatten() for x in  gfp_train["mutated_sequence"]])
y = gfp_train["DMS_score"]
ridge_model_oh.fit(X, y)


x_hat = ridge_model_oh.predict(X)
print(np.corrcoef(x_hat, y)

plt.scatter(x_hat, y)
plt.show()

# %%
np.corrcoef(x_hat, y)

# %%
len([st.seq_to_one_hot(x).flatten() for x in  gfp_train["mutated_sequence"]][0])

# %%
