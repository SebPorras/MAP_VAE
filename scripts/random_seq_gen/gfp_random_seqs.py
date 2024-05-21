import evoVAE.utils.seq_tools as st
import random
import pandas as pd


datasets = ["../data/gfp/GFP_AEQVI_full_04-29-2022_b08_ancestors_extants_no_syn_no_dupes.pkl",
            "../data/gfp/GFP_AEQVI_full_04-29-2022_b08_ancestors_no_syn_no_dupes.pkl",
            "../data/gfp/GFP_AEQVI_full_04-29-2022_b08_extants_no_syn_no_dupes.pkl"]


for dataset in datasets:

    ancestors = pd.read_pickle(dataset)
    seq_len = len(ancestors["sequence"].values[0])
    seqs = []
    ids = []
    count = 0
    for i in range(ancestors.shape[0]):
        rand_seq = "".join([st.IDX_TO_AA[random.randint(0, 20)] for x in range(seq_len)])
        seqs.append(rand_seq)
        ids.append(f"{i}_rand")
     

    df = pd.DataFrame({"id": ids, "sequence": seqs})
    encoding, weights = st.encode_and_weight_seqs(df['sequence'], 0.2)

    df['encoding'] = encoding
    df["weights"] = weights
    
    df.to_pickle(f"random_seqs_{dataset.split('/')[-1]}.pkl")

