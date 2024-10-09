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

import pandas as pd
import numpy as np
import src.utils.metrics as mt
import src.utils.seq_tools as st
import random

# pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import src.utils.statistics as stats
import src.utils.visualisation as vs
import yaml

# ### 5-fold validation hyperparamter tuning

# +
#### Model settings ####
# 1. A4: 7 dims, weight deacy: 0.001
# 2. GB1: 8 dims, weight deacy: 0.0001
# 3. GCN4: 5 dims, weight deacy: 0.005
# 4. GFP: 3 dims, weight deacy: 0.0005
# 5. MAFG: 4 dims, weight deacy: 0.00025

# +

all_data = pd.read_csv(
    "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/reconstruction_validation/k_fold_validation_marginal.csv"
)


datasets = ["gb1", "a4", "gcn4", "gfp", "mafg"]
for protein in datasets:
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 6)
    )  # Adjusted the figure size for better visibility
    subset = all_data[all_data["unique_id"].str.contains(protein)]
    subset = subset.sort_values(by="unique_id")

    subset["category"] = subset["unique_id"].apply(
        lambda x: (
            "ae"
            if "ae" in x.split("_")
            else (
                "a"
                if "a" in x.split("_")
                else (
                    "e"
                    if "e" in x.split("_")
                    else ("train" if "train" in x.split("_") else None)
                )
            )
        )
    )

    category_order = ["train", "ae", "a", "e"]
    x_tick_titles = [
        "Ancestor/Extant train",
        "Ancestor/Extant test",
        "Ancestor 5-fold val",
        "Extant 5-fold val",
    ]

    # # Create the barplot with seaborn
    bar = sns.barplot(
        x="category",
        y="marginal",
        data=subset,
        hue=subset["category"],
        palette=sns.color_palette("Set2"),
        ax=ax,
        order=category_order,
        dodge=False,
    )

    # Customize the plot
    ax.set_title(f"{protein.upper()}", fontsize=16)
    ax.set_xlabel("Training data", fontsize=12)
    ax.set_ylabel("mean log P(X)", fontsize=12)
    ax.set_xticklabels(x_tick_titles, fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        "Ancestors",
        "Ancestors/Extants test",
        "Extants",
        "Ancestors/Extants train",
    ]
    ax.legend(handles, new_labels, title=None)
    plt.title(f"{protein.upper()}")
    plt.show()
# -

# #### reconstruction metrics for val models

# +
thing = "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/reconstruction_validation/"
data = pd.read_csv(f"{thing}k_fold_covariances.csv")

palette = sns.color_palette()
palette = {
    "Ancestors": palette[0],
    "Extants": palette[2],
    "Ancestors/Extants": palette[1],
}

g = sns.catplot(
    data=data,
    x="family",
    y="pearson",
    hue="category",
    kind="bar",
    height=6,
    aspect=1.5,
    palette=palette,
    legend=False,
)

handles, labels = g.ax.get_legend_handles_labels()
print(labels)
g.ax.legend(
    handles,
    [
        "Ancestors (5-fold cross val)",
        "Extants (5-fold cross val)",
        "Ancestors/Extants (20% test split)",
    ],
    title="Training data",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)

g.set_axis_labels("Protein Family")
g.set_ylabels("Pearson correlation coefficient (r)")
# g.add_legend(title="Training data")
plt.title(
    "Validation MSA pariwise covariance correlation with reconstruction covariance (Validation models)"
)
plt.show()

# +
thing = "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/reconstruction_validation/"
data = pd.read_csv(f"{thing}k_fold_hamming.csv")

g = sns.catplot(
    data=data,
    x="family",
    y="hamming",
    hue="category",
    kind="bar",
    height=6,  # Adjust height here
    aspect=1.5,  # Adjust aspect ratio here
    legend=False,
)

handles, labels = g.ax.get_legend_handles_labels()

g.ax.legend(
    handles,
    [
        "Ancestors (5-fold cross val)",
        "Extants (5-fold cross val)",
        "Ancestors/Extants (20% test split)",
    ],
    title="Training data",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
g.set_axis_labels("Protein Family")
g.set_ylabels("Hamming distance")
# g.add_legend(title="Training data")
plt.title("Validation MSA hamming distance to reconstruction (Validation models)")
plt.show()
# -

# #### Checking how much models begin to overfit to the ancestor training data
#
# - split ancestors and extants into 5 folds.
# - hold out each k-fold for anc and extants
# - train on the remaining k-folds by combining ancestors and extants

data["family"] = data["family"].apply(lambda x: x.upper())
data.to_csv(
    "/Users/sebs_mac/uni_OneDrive/honours/data/overfit_testing/overfit_test_results.csv"
)

# +
data = pd.read_csv(
    "/Users/sebs_mac/uni_OneDrive/honours/data/overfit_testing/overfit_test_results.csv"
)

datasets = ["gb1", "a4", "gcn4", "gfp", "mafg"]
metrics = ["Training_ELBO", "Ancestor_ELBO", "Extant_ELBO"]
# Melt the DataFrame to long format
melted_data = pd.melt(
    data, id_vars=["family"], value_vars=metrics, var_name="metric", value_name="value"
)

sns.set_style("darkgrid")

# Plot with Seaborn
g = sns.catplot(
    data=melted_data,
    x="family",
    y="value",
    hue="metric",
    kind="bar",
    height=6,  # Adjust height here
    aspect=1.5,  # Adjust aspect ratio here
    legend=False,
)

g.set_axis_labels("Protein Family")
g.set_ylabels("ELBO")
plt.title(
    "Final ELBO on holdouts of extants or ancestors after training on Ancestors & extants"
)
g.add_legend()
plt.show()

# -

# ## Zero-shot tasks

# ### Full data

# +

all_data = pd.read_csv(
    "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/zero_shot_tasks/zero_shot_optimised_metrics.csv"
)
datasets = ["gb1", "a4", "gcn4", "gfp", "mafg"]
metrics = ["spearman_rho", "top_k_recall", "ndcg", "roc_auc"]
# Melt the DataFrame to long format
melted_data = pd.melt(
    all_data,
    id_vars=["family", "category"],
    value_vars=metrics,
    var_name="metric",
    value_name="value",
)
sns.set_style("darkgrid")
for metric in metrics:

    subset = melted_data[melted_data["metric"] == metric]
    # Plot with Seaborn
    g = sns.catplot(
        data=subset,
        x="family",
        y="value",
        hue="category",
        kind="bar",
        height=6,  # Adjust height here
        aspect=1.5,  # Adjust aspect ratio here
        legend=False,
    )

    g.set_axis_labels("Protein Family")
    g.set_titles("{col_name}")
    g.add_legend(title="Training data")
    plt.title(" ".join(metric.upper().split("_")))
    plt.show()


# -

# #### Curated data

# +

all_data = pd.read_csv(
    "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/curated/curated_metrics.csv"
)
melted_data = pd.melt(
    all_data,
    id_vars=["family", "category"],
    value_vars=metrics,
    var_name="metric",
    value_name="value",
)
sns.set_style("darkgrid")
metrics = ["spearman_rho", "top_k_recall", "ndcg", "roc_auc"]
for metric in metrics:

    subset = melted_data[melted_data["metric"] == metric]
    # Plot with Seaborn
    g = sns.catplot(
        data=subset,
        x="family",
        y="value",
        hue="category",
        kind="bar",
        height=6,  # Adjust height here
        aspect=1.5,  # Adjust aspect ratio here
        legend=False,
    )

    g.set_axis_labels("Protein Family")
    g.set_titles("{col_name}")
    # g.set(ylim=(0, None))
    g.add_legend(title="Training data")
    plt.title(" ".join(metric.upper().split("_")))
    plt.show()

# -

# #### Clustered data

# +
all_data = pd.read_csv(
    "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/clustering/clustering_metrics.csv"
)
datasets = ["gb1", "a4", "gcn4", "gfp", "mafg"]
train_size = [8000, 8000, 4000, 279, 2400]

metrics = ["spearman_rho", "top_k_recall", "ndcg", "roc_auc"]
for metric in metrics:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for protein, size in zip(datasets, train_size):

        subset = all_data[all_data["family"] == protein.upper()]
        subset = subset.sort_values(by="extant")

        sns.scatterplot(x="extant", y=metric, data=subset, ax=ax, marker="o")

        sns.lineplot(
            x="extant",
            y=metric,
            data=subset,
            ax=ax,
            label=f"{protein.upper()} ({size} seqs)",
        )

    plt.title(metric)
    plt.legend(title="Protein (Training size)")

    xticks = np.arange(0, all_data["extant"].max() + 0.05, 0.05)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x*100)}%" for x in xticks])
    ax.set_xlabel("Percentage of extants in sample")

    plt.show()

# -

# ## Covariances (column reconstruction)

# ### Zero-shot models
#
# - trained on whole dataset from 5 random starts
# ### DATA LEAKAGE IN THIS BEACUSE THEY HAVE SEEN THE EXTANTS BEFORE

# +

covariances = pd.read_csv(
    "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/zero_shot_tasks/pairwise_covariance.csv"
)
covariances
g = sns.catplot(
    data=covariances,
    x="family",
    y="pearson",
    hue="category",
    kind="bar",
    height=6,  # Adjust height here
    aspect=1.5,  # Adjust aspect ratio here
    legend=False,
)

g.set_axis_labels("Protein Family")
g.set_ylabels("Pearson correlation coefficient (r)")
g.add_legend(title="Training data")
plt.title(
    "Extant MSA pariwise covariance correlation with reconstruction covariance (zero shot models)"
)
plt.show()
# -

# #### Curated data

# +

covariances = pd.read_csv(
    "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/curated/curated_extant_covariances.csv"
)
covariances
g = sns.catplot(
    data=covariances,
    x="family",
    y="pearson",
    hue="category",
    kind="bar",
    height=6,  # Adjust height here
    aspect=1.5,  # Adjust aspect ratio here
    legend=False,
)

g.set_axis_labels("Protein Family")
g.set_ylabels("Pearson correlation coefficient (r)")
g.add_legend(title="Training data")
plt.title(
    "Extant MSA pariwise covariance correlation with reconstruction covariance (Curated models)"
)
plt.show()
# -

# ### Clustered models
# - Sample size controlled, only the number of extants is changed

# +

vars = [
    "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/clustering/clustering_extant_covariances.csv",
    "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/clustering/clustering_validation_covariances.csv",
]

titles = ["extants", "validation set"]


for v, t in zip(vars, titles):

    all_data = pd.read_csv(v)
    datasets = ["gb1", "gcn4", "gfp", "mafg", "a4"]  # List of protein families
    train_size = [8000, 4000, 279, 2400, 8000]
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 6)
    )  # Create a single figure for all lines

    # Iterate through each dataset and plot them on the same axes
    for protein, size in zip(datasets, train_size):
        subset = all_data[all_data["family"] == protein.upper()]
        subset = subset.sort_values(by="extant_proportion")
        sns.scatterplot(
            x="extant_proportion", y="pearson", data=subset, ax=ax, marker="o"
        )

        sns.lineplot(
            x="extant_proportion",
            y="pearson",
            data=subset,
            ax=ax,
            label=f"{protein.upper()} ({size} seqs)",
        )

    # Set the title and legend
    plt.title(
        f"Pearson correlation of pairwise covariances between {t} and reconstructions"
    )
    plt.legend(title="Protein Family")

    xticks = np.arange(0, all_data["extant_proportion"].max() + 0.05, 0.05)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x*100)}%" for x in xticks])
    ax.set_xlabel("Percentage of extants in sample")

    ax.set_ylabel("Pearson correlation coefficient (r)")

    plt.show()


# -

# ## Hamming distances (row reconstruction)

# ### Clustering

# +

vars = [
    "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/clustering/clustering_extant_hamming.csv",
    "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/clustering/clustering_validation_hamming.csv",
]

titles = ["extants", "validation set"]


for v, t in zip(vars, titles):

    all_data = pd.read_csv(v)

    datasets = ["gb1", "gcn4", "gfp", "mafg", "a4"]  # List of protein families
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 6)
    )  # Create a single figure for all lines

    # Iterate through each dataset and plot them on the same axes
    for protein in datasets:
        subset = all_data[all_data["family"] == protein.upper()]
        subset = subset.sort_values(by="extant_proportion")
        sns.scatterplot(
            x="extant_proportion", y="hamming_distance", data=subset, ax=ax, marker="o"
        )

        sns.lineplot(
            x="extant_proportion",
            y="hamming_distance",
            data=subset,
            ax=ax,
            label=protein.upper(),
        )

    # Set the title and legend
    plt.title(f"Hamming distance between {t} and reconstructions")
    plt.legend(title="Protein Family")

    xticks = np.arange(0, all_data["extant_proportion"].max() + 0.05, 0.05)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x*100)}%" for x in xticks])
    ax.set_xlabel("Percentage of extants in sample")
    ax.set_xticklabels([f"{int(x*100)}%" for x in xticks])
    ax.set_xlabel("Percentage of extants in sample")

    plt.show()


# -

# ## Latent space representation

# #### Tree visualisations

# +

protein = ["gb1", "a4", "gcn4", "gfp", "mafg"]
tree_id = [1, 1, 0, 1, 0]
data = ["ae", "a", "e"]
labels = ["Ancestor/Extant", "Ancestor", "Extant"]
wts = [
    "SPG1_STRSG/1-448",
    "A4_HUMAN/1-770",
    "GCN4_YEAST/1-281",
    "GFP_AEQVI/1-238",
    "MAFG_MOUSE/1-41",
]

with open("../data/dummy_config.yaml", "r") as stream:
    settings = yaml.safe_load(stream)

sns.set_style("darkgrid")
for (
    p,
    id,
    wt,
) in zip(protein, tree_id, wts):
    for d, lab in zip(data, labels):
        state_dict = f"/Users/sebs_mac/uni_OneDrive/honours/data/vis_models/{p}_{d}_r1_model_state.pt"
        tree_seqs = (
            f"/Users/sebs_mac/uni_OneDrive/honours/data/vis_models/{p}_tree_{id}.aln"
        )

        vs.vis_tree(
            wt, tree_seqs, state_dict, settings, f"{p.upper()} - {lab} model", rgb=True
        )
        vs.vis_tree(
            wt,
            tree_seqs,
            state_dict,
            settings,
            f"{p.upper()} - {lab} model",
            rgb=True,
            lower_2d=True,
        )

    plt.show()
# -

# #### Variant visualisations
#
# The following plots are generated from models created using the optimised paramters

# +
protein = ["a4", "gcn4", "gfp", "mafg", "gb1"]
data = ["ae", "a", "e"]
labels = ["Ancestor/Extant", "Ancestor", "Extant"]
wts = [
    "A4_HUMAN/1-770",
    "GCN4_YEAST/1-281",
    "GFP_AEQVI/1-238",
    "MAFG_MOUSE/1-41",
    "SPG1_STRSG/1-448",
]
latent_dims = [7, 5, 3, 4, 8]
fracs = [0.01, 0.01, 0.01, 0.01, 0.01]
tree_id = [1, 0, 1, 0, 1]
variants = [
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/A4_HUMAN_Seuma_2022.csv",
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GCN4_YEAST_Staller_2018.csv",
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv",
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/MAFG_MOUSE_Tsuboyama_2023_1K1V.csv",
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/SPG1_STRSG_Wu_2016.csv",
]

sns.set_style("darkgrid")
for p, wt, v, frac, id, dims in zip(
    protein, wts, variants, fracs, tree_id, latent_dims
):
    for (
        d,
        lab,
    ) in zip(data, labels):

        with open("../data/dummy_config.yaml", "r") as stream:
            settings = yaml.safe_load(stream)
        settings["latent_dims"] = dims

        state_dict = f"/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/zero_shot_tasks/zero_shot_1/model_states/{p}_{d}_r1_model_state.pt"
        tree_seqs = (
            f"/Users/sebs_mac/uni_OneDrive/honours/data/vis_models/{p}_tree_{id}.aln"
        )
        vs.visualise_variants(
            settings,
            v,
            state_dict,
            f"{p.upper()} - {lab} model",
            tree_seqs,
            wt,
            vis_2D=True,
            frac=frac,
        )


# -

# We can do the same but for the models trained with only 3 dimensions so it's a more direct visualisation

# +
protein = ["a4", "gcn4", "gfp", "mafg", "gb1"]
data = ["ae", "a", "e"]
labels = ["Ancestor/Extant", "Ancestor", "Extant"]
wts = [
    "A4_HUMAN/1-770",
    "GCN4_YEAST/1-281",
    "GFP_AEQVI/1-238",
    "MAFG_MOUSE/1-41",
    "SPG1_STRSG/1-448",
]
tree_id = [1, 0, 1, 0, 1]
variants = [
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/A4_HUMAN_Seuma_2022.csv",
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GCN4_YEAST_Staller_2018.csv",
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv",
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/MAFG_MOUSE_Tsuboyama_2023_1K1V.csv",
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/SPG1_STRSG_Wu_2016.csv",
]
fracs = [0.01, 0.01, 0.01, 0.01, 0.01]
with open("../data/dummy_config.yaml", "r") as stream:
    settings = yaml.safe_load(stream)
# sns.set_style("darkgrid")

for p, wt, v, frac, id in zip(protein, wts, variants, fracs, tree_id):
    for (
        d,
        lab,
    ) in zip(data, labels):

        # use this to get the WT
        tree_seqs = (
            f"/Users/sebs_mac/uni_OneDrive/honours/data/vis_models/{p}_tree_{id}.aln"
        )
        state_dict = state_dict = (
            f"/Users/sebs_mac/uni_OneDrive/honours/data/vis_models/{p}_{d}_r1_model_state.pt"
        )
        test = vs.visualise_variants(
            settings,
            v,
            state_dict,
            f"{p.upper()} - {lab} model",
            tree_seqs,
            wt,
            vis_2D=False,
            frac=frac,
        )


# -

# # Supervised learning

# +

data = pd.read_csv(
    "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/zero_shot_tasks/top_models/all_top_model.csv"
)

# uncomment to only see VAE and LASE
# data = data[(data["category"] != "ESM2") & (data["category"] != "One-Hot")]

metrics = ["spearman", "k_recall", "ndcg", "roc_auc"]
top_models = ["Ridge", "LassoLars"]

for model in top_models:
    for m in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        data_type = ["train", "test"]

        for idx, t in enumerate(data_type):
            sns.barplot(
                data=data[(data["model"] == model) & (data["train_set"] == t)],
                x="family",
                y=m,
                hue="category",
                ax=axes[idx],
            )

            axes[idx].legend().remove()
            axes[idx].set_title(f"{t.capitalize()} split")
            axes[idx].set_xlabel("Protein Family")
            axes[idx].set_ylabel(m.capitalize())

        handles, labels = axes[idx].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

        fig.suptitle(f"Representations with {model} top model")
        plt.tight_layout()
        plt.show()
# -

# # Ancestor to extant sequence composition

# +


aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/all_alns/"
proteins = ["gb1", "gcn4", "gfp", "mafg", "a4"]
mutation_positions = [[264, 265, 266, 279], [101, 144], [3, 237], [0, 40], [671, 712]]
ancestor_counts = [12733, 11623, 497, 3682, 29940]
extant_counts = [2286, 331, 324, 962, 4936]

for prot, mutations, anc, ext in zip(
    proteins, mutation_positions, ancestor_counts, extant_counts
):

    extants = st.read_aln_file(aln_path + f"{prot.upper()}_extants_no_dupes.aln")
    ancestors = st.read_aln_file(aln_path + f"{prot.upper()}_ancestors_no_dupes.aln")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    vs.plot_entropy(
        ancestors=ancestors,
        extants=extants,
        anc_count=anc,
        ext_count=ext,
        mutations=mutations,
        protein=prot.upper(),
        title="Column entropy ancestors vs extants",
        ax=ax1,
    )

    vs.plot_ppm_difference(extants, ancestors, fig, ax2)
    ax2.set_title(f"{prot.upper()} Extant PPM - Ancestor PPM ")
    plt.show()
# -

# ## Determining similarity of ancestors to extants

# +


fig, ax = plt.subplots(1, 1, figsize=(12, 8))
bar = sns.barplot(
    x="experiment",
)

# +
data = pd.read_csv(
    "/Users/sebs_mac/uni_OneDrive/honours/data/overfit_testing/all_embedding_closest_results.csv"
)
data = data[
    ["experiment", "Ancestor_to_Ancestor", "Ancestor_to_Extant", "Extant_to_Extant"]
]
datasets = ["gb1", "a4", "gcn4", "gfp", "mafg"]

metrics = ["Ancestor_to_Ancestor", "Ancestor_to_Extant", "Extant_to_Extant"]
# Melt the DataFrame to long format
melted_data = pd.melt(
    data,
    id_vars=["experiment"],
    value_vars=metrics,
    var_name="metric",
    value_name="value",
)
melted_data
sns.set_style("darkgrid")

# Plot with Seaborn
g = sns.catplot(
    data=melted_data,
    x="experiment",
    y="value",
    hue="metric",
    kind="bar",
    height=6,  # Adjust height here
    aspect=1.5,  # Adjust aspect ratio here
    legend=False,
)

g.set_axis_labels("Protein Family")
g.set_ylabels("proportion")
plt.title(
    "Proportion of closest sequence pairs using ESM2 embeddings and cosine similarity"
)
g.add_legend()
plt.show()

# -
# ## Using a single tree to train


# +
data = pd.read_csv(
    "/Users/sebs_mac/uni_OneDrive/honours/data/single_tree/output/single_tree_zero_shot_metrics.csv"
)
metrics = ["spearman_rho", "top_k_recall", "ndcg", "roc_auc"]
melted_data = pd.melt(
    data,
    id_vars=["Protein_family", "data"],
    value_vars=metrics,
    var_name="metric",
    value_name="value",
)

for metric in metrics:
    subset = melted_data[melted_data["metric"] == metric]
    g = sns.catplot(
        data=subset,
        x="Protein_family",
        y="value",
        hue="data",
        kind="bar",
        height=6,  # Adjust height here
        aspect=1.5,  # Adjust aspect ratio here
        legend=False,
    )
    g.set_axis_labels("Protein Family")
    g.set_titles("{col_name}")
    # g.set(ylim=(0, None))
    g.add_legend(title="Training data")
    plt.title(" ".join(metric.upper().split("_")))
    plt.show()
# -

data.head()
