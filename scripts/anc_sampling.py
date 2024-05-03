import evoVAE.utils.seq_tools as st
import pandas as pd
import os, sys


#R1

sizes = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]

for size in sizes:

    sampled_ancestors = st.sample_ancestor_trees(tree_count=100, rejection_threshold=size, invalid_trees=[11, 31, 48, 60, 78, 63],
                                          tree_path="/scratch/user/s4646506/gfp_alns/ancestor_trees/",
                                          ancestor_pkl_path="/scratch/user/s4646506/evoVAE/data/GFP_AEQVI_full_04-29-2022_b08_ancestors_no_syn.pkl")
    weights = st.reweight_sequences(sampled_ancestors["sequence"].to_numpy(), 0.2)
    sampled_ancestors['weights'] = weights
    sampled_ancestors.to_pickle(f"GFP_AEQVI_encoded_weighted_{size}_rejection_r{sys.argv[1]}_ancestors_no_syn.pkl")



