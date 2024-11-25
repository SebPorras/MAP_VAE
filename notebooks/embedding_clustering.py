# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

import igraph as ig
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import MAP_VAE.utils.seq_tools as st

# +


results = pd.read_csv(
    f"/Users/sebs_mac/uni_OneDrive/honours/data/all_alns/clusters/gcn4_an_ex_cluster.tsv",
    sep="\t",
)
mark_ancestors = lambda x: 1 if "tree" in x else 0
is_ancestor = results["sequence"].apply(mark_ancestors)
results["is_ancestor"] = is_ancestor
representative_ids = results["cluster"].unique()
clusters = [
    results.loc[results["cluster"] == cluster] for cluster in representative_ids
]
representative_ids.shape
# -

results.loc[results["is_ancestor"] == 0].shape

# +
import random

SAMPLE_SIZE = 200
extant_proportion = 0.5

# read in the embeddings
path = "/Users/sebs_mac/uni_OneDrive/honours/data/all_alns/embeddings/"
ems = pd.read_pickle(path + "gcn4_ancestors_extants_no_dupes_embeddings.pkl")

random.seed(42)
num_experiments = 1000

anc_to_anc = 0
ext_to_anc = 0
ext_to_ext = 0


for i in range(num_experiments):

    sample_ids = st.sample_extant_ancestors(
        clusters, SAMPLE_SIZE, extant_proportion=extant_proportion
    )
    subset = ems[ems["id"].isin(sample_ids)]

    # store the names for access later
    idx_to_name = {idx: name for idx, name in enumerate(subset["id"])}
    name_to_idx = {name: idx for idx, name in enumerate(subset["id"])}

    embeddings = np.array([x.numpy() for x in subset["embeddings"]])
    distances = cosine_similarity(embeddings, embeddings)

    for i in range(distances.shape[0]):
        # don't find self as closest
        distances[i, i] = -np.inf
        closest_idx = np.argmax(distances[i, :])
        query = idx_to_name[i]
        closest = idx_to_name[closest_idx]

        if "tree" in query and "tree" in closest:

            anc_to_anc += 1
        elif "tree" in query and "tree" not in closest:
            ext_to_anc += 1

        elif "tree" not in query and "tree" in closest:
            ext_to_anc += 1

        elif "tree" not in query and "tree" not in closest:
            ext_to_ext += 1

anc_to_anc /= num_experiments * SAMPLE_SIZE
ext_to_anc /= num_experiments * SAMPLE_SIZE
ext_to_ext /= num_experiments * SAMPLE_SIZE

print(anc_to_anc, ext_to_anc, ext_to_ext)


# +
import random

SAMPLE_SIZE = 1000
extant_proportion = 0.5
protein = "mafg"
results = pd.read_csv(
    f"/Users/sebs_mac/uni_OneDrive/honours/data/all_alns/clusters/{protein}_an_ex_cluster.tsv",
    sep="\t",
)
mark_ancestors = lambda x: 1 if "tree" in x else 0
is_ancestor = results["sequence"].apply(mark_ancestors)
results["is_ancestor"] = is_ancestor
representative_ids = results["cluster"].unique()
clusters = [
    results.loc[results["cluster"] == cluster] for cluster in representative_ids
]


# read in the embeddings
path = "/Users/sebs_mac/uni_OneDrive/honours/data/all_alns/embeddings/"
ems = pd.read_pickle(path + f"{protein}_ancestors_extants_no_dupes_embeddings.pkl")

random.seed(42)
num_experiments = 500

anc_to_anc = 0
ext_to_anc = 0
ext_to_ext = 0

rand_anc_to_anc = 0
rand_ext_to_anc = 0
rand_ext_to_ext = 0

for i in range(num_experiments):

    sample_ids = st.sample_extant_ancestors(
        clusters, SAMPLE_SIZE, extant_proportion=extant_proportion
    )

    subset = ems[ems["id"].isin(sample_ids)]

    # store the names for access later
    idx_to_name = {idx: name for idx, name in enumerate(subset["id"])}
    name_to_idx = {name: idx for idx, name in enumerate(subset["id"])}

    embeddings = np.array([x.numpy() for x in subset["embeddings"]])
    rand = np.random.rand(embeddings.shape[0], embeddings.shape[1])
    distances = cosine_similarity(embeddings, embeddings)
    rand_distances = cosine_similarity(rand, rand)

    for i in range(distances.shape[0]):
        # don't find self as closest
        distances[i, i] = -np.inf
        closest_idx = np.argmax(distances[i, :])
        query = idx_to_name[i]
        closest = idx_to_name[closest_idx]

        if "tree" in query and "tree" in closest:

            anc_to_anc += 1
        elif "tree" in query and "tree" not in closest:
            ext_to_anc += 1

        elif "tree" not in query and "tree" in closest:
            ext_to_anc += 1

        elif "tree" not in query and "tree" not in closest:
            ext_to_ext += 1

    for i in range(rand_distances.shape[0]):
        # don't find self as closest
        rand_distances[i, i] = -np.inf
        closest_idx = np.argmax(rand_distances[i, :])
        query = idx_to_name[i]
        closest = idx_to_name[closest_idx]

        if "tree" in query and "tree" in closest:

            rand_anc_to_anc += 1
        elif "tree" in query and "tree" not in closest:
            rand_ext_to_anc += 1

        elif "tree" not in query and "tree" in closest:
            rand_ext_to_anc += 1

        elif "tree" not in query and "tree" not in closest:
            rand_ext_to_ext += 1

rand_anc_to_anc /= num_experiments * SAMPLE_SIZE
rand_ext_to_anc /= num_experiments * SAMPLE_SIZE
rand_ext_to_ext /= num_experiments * SAMPLE_SIZE

anc_to_anc /= num_experiments * SAMPLE_SIZE
ext_to_anc /= num_experiments * SAMPLE_SIZE
ext_to_ext /= num_experiments * SAMPLE_SIZE

print(anc_to_anc, ext_to_anc, ext_to_ext)


print(rand_anc_to_anc, rand_ext_to_anc, rand_ext_to_ext)


# +
# standard - no mmseq clustering

import random

SAMPLE_SIZE = 500
extant_proportion = 0.5
protein = "gb1"

# read in the embeddings
path = "/Users/sebs_mac/uni_OneDrive/honours/data/all_alns/embeddings/"
ems = pd.read_pickle(path + f"{protein}_ancestors_extants_no_dupes_embeddings.pkl")
mark_ancestors = lambda x: 1 if "tree" in x else 0
is_ancestor = ems["id"].apply(mark_ancestors)
ems["is_ancestor"] = is_ancestor


random.seed(42)
num_experiments = 1000

anc_to_anc = 0
ext_to_anc = 0
ext_to_ext = 0

rand_anc_to_anc = 0
rand_ext_to_anc = 0
rand_ext_to_ext = 0

for i in range(num_experiments):

    extants = ems[ems["is_ancestor"] == 0].sample(
        n=int(SAMPLE_SIZE / 2), random_state=42
    )
    ancestors = ems[ems["is_ancestor"] == 1].sample(
        n=int(SAMPLE_SIZE / 2), random_state=42
    )
    subset = pd.concat([extants, ancestors])

    # store the names for access later
    idx_to_name = {idx: name for idx, name in enumerate(subset["id"])}
    name_to_idx = {name: idx for idx, name in enumerate(subset["id"])}

    embeddings = np.array([x.numpy() for x in subset["embeddings"]])
    rand = np.random.rand(embeddings.shape[0], embeddings.shape[1])
    distances = cosine_similarity(embeddings, embeddings)
    rand_distances = cosine_similarity(rand, rand)

    for i in range(distances.shape[0]):
        # don't find self as closest
        distances[i, i] = -np.inf
        closest_idx = np.argmax(distances[i, :])
        query = idx_to_name[i]
        closest = idx_to_name[closest_idx]

        if "tree" in query and "tree" in closest:

            anc_to_anc += 1
        elif "tree" in query and "tree" not in closest:
            ext_to_anc += 1

        elif "tree" not in query and "tree" in closest:
            ext_to_anc += 1

        elif "tree" not in query and "tree" not in closest:
            ext_to_ext += 1

    for i in range(rand_distances.shape[0]):
        # don't find self as closest
        rand_distances[i, i] = -np.inf
        closest_idx = np.argmax(rand_distances[i, :])
        query = idx_to_name[i]
        closest = idx_to_name[closest_idx]

        if "tree" in query and "tree" in closest:

            rand_anc_to_anc += 1
        elif "tree" in query and "tree" not in closest:
            rand_ext_to_anc += 1

        elif "tree" not in query and "tree" in closest:
            rand_ext_to_anc += 1

        elif "tree" not in query and "tree" not in closest:
            rand_ext_to_ext += 1

rand_anc_to_anc /= num_experiments * SAMPLE_SIZE
rand_ext_to_anc /= num_experiments * SAMPLE_SIZE
rand_ext_to_ext /= num_experiments * SAMPLE_SIZE

anc_to_anc /= num_experiments * SAMPLE_SIZE
ext_to_anc /= num_experiments * SAMPLE_SIZE
ext_to_ext /= num_experiments * SAMPLE_SIZE

print(anc_to_anc, ext_to_anc, ext_to_ext)
print(rand_anc_to_anc, rand_ext_to_anc, rand_ext_to_ext)

