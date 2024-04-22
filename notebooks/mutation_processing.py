# %%

import pandas as pd
import os
import evoVAE.utils.seq_tools as st
import evoVAE.utils.metrics as mt
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 30)
os.getcwd()

# %% [markdown]
# This notebook just demonstrates the basic use of different summary statistics to make sure they were working on random data

# %%
metadata = pd.read_csv("../data/dms_data/DMS_substitutions.csv")
test = ["GFP", "SPG1_STRSG_Wu_2016", "SPG1_STRSG_Olson_2014"]
metadata = metadata[
    metadata["DMS_id"].str.contains("GFP")
    | metadata["DMS_id"].str.contains("SPG1_STRSG_Wu_2016")
]
metadata

# %%
dms_gfp = pd.read_csv("../data/dms_data/GFP_AEQVI_Sarkisyan_2016.csv")
aln_gfp = st.read_aln_file("../data/dms_data/GFP_AEQVI_full_04-29-2022_b08.a2m")
metadata = pd.read_csv("../data/dms_data/DMS_substitutions.csv")
metadata = metadata[metadata["DMS_id"] == "GFP_AEQVI_Sarkisyan_2016"]


# %%
dms_gfp

# %%

subset_dms = dms_gfp[:100].copy()
preds = np.random.uniform(-100, -10, size=100)
# subset_dms = subset_dms.loc[subset_dms['DMS_score'].sort_values(ascending=False).index]
# preds = np.arange(100, 0, -1)
subset_dms["predictions"] = preds
subset_dms

# %%
subset_dms["mutant"]

# %%

spear_rho, k_recall, ndcg, roc_auc = mt.summary_stats(
    subset_dms["predictions"], subset_dms["DMS_score"], subset_dms["DMS_score_bin"]
)
spear_rho, k_recall, ndcg, roc_auc

# %%
