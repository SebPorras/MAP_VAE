import evoVAE.utils.seq_tools as st
import pandas as pd
import os, sys


# R1

FILE_NAME = ""
TREE_PATH = "/scratch/user/s4646506/gb1/ancestor_trees/"
INVALID_TREES = [3, 4, 5, 6, 9, 10, 12, 15, 16, 17, 18, 19, 23, 25, 26, 27, 29, 30]
ALL_ANC_PICKLE = "/scratch/user/s4646506/evoVAE/data/gb1/gb1_ancestors_encoded_weighted_no_dupes.pkl"
REPLICATES = 5
sizes = [0.10, 0.18, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
sizes = [0.9]

"""

for size in sizes:

    sampled_ancestors = st.sample_ancestor_trees(
        tree_count=30,
        rejection_threshold=size,
        invalid_trees=INVALID_TREES,
        tree_path=TREE_PATH,
        ancestor_pkl_path=ALL_ANC_PICKLE,
    )

    weights = st.reweight_by_seq_similarity(
        sampled_ancestors["sequence"].to_numpy(), 0.2
    )
    sampled_ancestors["weights"] = weights
    sampled_ancestors.to_pickle(
        f"gb1_ancestors_encoded_weighted_{size}_rejection_r{sys.argv[1]}.pkl"
    )
"""
    

def thing(args):

    sample_size, r, no_dupes = args

    subset = no_dupes.sample(frac=sample_size, replace=False, random_state=r)

    weights = st.reweight_by_seq_similarity(
        subset["sequence"].to_numpy(), 0.2
    )
    subset["weights"] = weights

    subset.to_pickle(
        f"gb1_ancestors_no_dupes_encoded_weighted_{sample_size}_sample_r{r}.pkl"
    )


for sample_size in sizes:

    r = 2

    no_dupes = pd.read_pickle(ALL_ANC_PICKLE)
    [thing((sample_size, r, no_dupes)) for sample_size in sizes]
    
       
