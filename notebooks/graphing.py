# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: embed
#     language: python
#     name: python3
# ---

import MAP_VAE.utils.visualisation as vs
import MAP_VAE.utils.seq_tools as st
import yaml
from MAP_VAE.models.seqVAE import SeqVAE
from MAP_VAE.utils.datasets import MSA_Dataset
import pandas as pd
import torch
import numpy as np
import igraph as ig
from igraph import Graph

with open("../data/dummy_config.yaml", "r") as stream:
    settings = yaml.safe_load(stream)

# Cassowary

# +
path = "/Users/sebs_mac/uni_OneDrive/honours/data/cassowary/vis/"
cass_tree = path + "tree_1_ancestors_extants.fasta"
a_state_dict = "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/cassowary_standard/cassowary_a_r1/cassowary_a_r1_model_state.pt"
e_state_dict = "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/cassowary_standard/cassowary_e_r1/cassowary_e_r1_model_state.pt"


vs.vis_tree(
    None,
    cass_tree,
    a_state_dict,
    settings,
    "RNAseZ - Ancestor model",
    rgb=True,
    lower_2d=True,
)
vs.vis_tree(
    None,
    cass_tree,
    e_state_dict,
    settings,
    "RNAseZ - Extant model",
    rgb=True,
    lower_2d=True,
)
vs.vis_tree(
    None, cass_tree, a_state_dict, settings, "RNAseZ - Ancestor model", rgb=True
)
vs.vis_tree(None, cass_tree, e_state_dict, settings, "RNAseZ - Extant model", rgb=True)

# +
vs.latent_tree_to_itol(
    "RNAseZ_ancestor_model",
    state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/cassowary_standard/cassowary_a_r1/cassowary_a_r1_model_state.pt",
    tree_seq_path=cass_tree,
    settings=settings,
)

vs.latent_tree_to_itol(
    "RNAseZ_extant_model",
    state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/cassowary_standard/cassowary_e_r1/cassowary_e_r1_model_state.pt",
    tree_seq_path=cass_tree,
    settings=settings,
)

# +

tree = st.read_aln_file(cass_tree)
tree
one_hot = tree["sequence"].apply(st.seq_to_one_hot)
tree["encoding"] = one_hot

device = torch.device("mps")
tree_dataset = MSA_Dataset(
    tree["encoding"],
    pd.Series(np.arange(len(one_hot))),
    tree["id"],
    device=device,
)

num_seqs = len(tree["sequence"])
tree_loader = torch.utils.data.DataLoader(
    tree_dataset, batch_size=num_seqs, shuffle=False
)

seq_len = tree_dataset[0][0].shape[0]
input_dims = seq_len * settings["AA_count"]

model = SeqVAE(
    dim_latent_vars=settings["latent_dims"],
    dim_msa_vars=input_dims,
    num_hidden_units=settings["hidden_dims"],
    settings=settings,
    num_aa_type=settings["AA_count"],
)

state_dict = "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/cassowary_standard/cassowary_a_r1/cassowary_a_r1_model_state.pt"
model.load_state_dict(torch.load(state_dict, map_location=device))
model = model.to(device)

# -

model.eval()
ids = []
mus = []
sigmas = []
with torch.no_grad():
    for x, _, name in tree_loader:
        x = torch.flatten(x, start_dim=1)
        mu, sigma = model.encoder(x)
        mus.extend(mu.cpu().numpy())
        sigmas.extend(sigma.cpu().numpy())
        ids.extend(name)


coordinates = pd.DataFrame({"id": ids, "mu": mus, "sigma": sigmas})
coordinates


def wasserstein_2_distance(mu1: float, sig1: float, mu2: float, sig2: float) -> float:
    """
    Calculates the Wasserstein-2 distance for 2 Gaussians.

    Uses the closed form determined by Givens and Shortt, 1984.
    https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/

    Assumes that these are 1D Gaussians so standard deviation is given rather than the
    covaraince matrix.

    Returns:
    Wasserstein-2 distance - note this is a squared value
    """

    squared_l2_norm = (mu1 - mu2) ** 2

    cov_term = sig1**2 + sig2**2 - 2 * ((sig1 * sig2**2 * sig1) ** 0.5)

    wasserstein = squared_l2_norm + cov_term

    return wasserstein


# +
ids_to_idx = {id: idx for idx, id in enumerate(coordinates["id"])}
idx_to_ids = {idx: id for idx, id in enumerate(coordinates["id"])}

dist_mat = np.zeros((len(ids_to_idx), len(ids_to_idx)))
print(dist_mat.shape)

for i in range(dist_mat.shape[0] - 1):
    for j in range(i + 1, dist_mat.shape[0]):

        dist = np.sqrt(
            np.sum(
                [
                    wasserstein_2_distance(mu_i, sig_i, mu_j, sig_j)
                    for mu_i, sig_i, mu_j, sig_j in zip(
                        mus[i], sigmas[i], mus[j], sigmas[j]
                    )
                ]
            )
        )
        # dist = np.mean([wasserstein_2_distance(mu_i, sig_i, mu_j, sig_j) for mu_i, sig_i, mu_j, sig_j in zip(mus[i], sigmas[i],  mus[j], sigmas[j])])
        dist_mat[i, j] = dist_mat[j, i] = dist
# -

plt.figure(figsize=(10, 8))  # Set the figure size
plt.imshow(dist_mat, cmap="viridis")
plt.colorbar()  # Add a colorbar to show the scale
plt.show()

g = Graph.Weighted_Adjacency(dist_mat.tolist(), mode="undirected", attr="weight")

print("test")
commmunities = g.community_walktrap(weights=g.es["weight"])
for_plot = commmunities.as_clustering()

# +

for i, community in enumerate(for_plot):
    print(f"Community {i}:")
    for v in community:
        print(v)

# +
import igraph as ig
import matplotlib.pyplot as plt

g = ig.Graph.Famous("Zachary")
communities = g.community_edge_betweenness()
communities = communities.as_clustering()
num_communities = len(communities)
palette = ig.RainbowPalette(n=num_communities)
for i, community in enumerate(communities):
    g.vs[community]["color"] = i
    community_edges = g.es.select(_within=community)
    community_edges["color"] = i

fig, ax = plt.subplots()

ig.plot(
    communities,
    palette=palette,
    edge_width=1,
    target=ax,
    vertex_size=20,
)

# Create a custom color legend
legend_handles = []
for i in range(num_communities):
    handle = ax.scatter(
        [],
        [],
        s=100,
        facecolor=palette.get(i),
        edgecolor="k",
        label=i,
    )
    legend_handles.append(handle)
ax.legend(
    handles=legend_handles,
    title="Community:",
    bbox_to_anchor=(0, 1.0),
    bbox_transform=ax.transAxes,
)
plt.show()

# -
import os

path = "/Users/sebs_mac/uni_OneDrive/honours/data/single_tree/output/recon_metrics/"
files = [pd.read_csv(path + x) for x in os.listdir(path) if x.endswith(".csv")]
files[0]


all_data = pd.concat(files, ignore_index=True)
all_data.head()

protein_family = [
    x.upper() for x in all_data["unique_id"].str.split("_", expand=True)[0]
]
type = [
    "Extant" if x == "ext" else "Ancestor"
    for x in all_data["unique_id"].str.split("_", expand=True)[1]
]
all_data["Protein_family"] = protein_family
all_data["data"] = type
all_data
all_data.to_csv(
    "/Users/sebs_mac/uni_OneDrive/honours/data/single_tree/output/single_tree_marginal_metrics.csv",
    index=False,
)
