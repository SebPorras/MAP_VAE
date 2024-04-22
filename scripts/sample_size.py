import evoVAE.utils.seq_tools as st
import pandas as pd
import numpy as np

DATA_PATH = "/scratch/user/s4646506/gfp_alns/independent_runs/"
ancestors: pd.DataFrame = pd.read_pickle(
    DATA_PATH + "GFP_AEQVI_full_04-29-2022_b08_ancestors_extants_no_syn.pkl"
)

fracs = np.arange(0.05, 1, 0.05)

for idx, frac in enumerate(fracs):

    sub_sample = ancestors.sample(frac=frac, replace=False)
    mean_dev = st.population_profile_deviation(ancestors, sub_sample)

    print(f"{frac},{mean_dev}")
