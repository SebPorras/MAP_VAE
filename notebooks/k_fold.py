# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: embed
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import MAP_VAE.utils.metrics as mt
import MAP_VAE.utils.seq_tools as st
import random

# pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import random
import os


# ### GB1

# +
path = "/Users/sebs_mac/k_fold_results/gb1/"

metric_paths = [
    "_".join(subdir.split("_")[:2]) + "_5_fold_" + "_".join(subdir.split("_")[2:])
    for subdir in os.listdir(path)
    if subdir != ".DS_Store"
]
metrics = [
    pd.read_csv(
        f"{path}{'_'.join(subdir.split('_')[:2] + subdir.split('_')[-2:])}/{subdir}_r1_metrics.csv"
    )
    for subdir in metric_paths
]
all_metrics = pd.concat(metrics)
dataset = [x[1] for x in all_metrics["unique_id"].str.split("_")]
decay = [x[5] for x in all_metrics["unique_id"].str.split("_")]
all_metrics["dataset"] = dataset
all_metrics["weight_decay"] = decay

gb1_ae = all_metrics[all_metrics["dataset"] == "ae"]
gb1_a = all_metrics[all_metrics["dataset"] == "a"]
gb1_e = all_metrics[all_metrics["dataset"] == "e"]


# +


def summary_data(data: pd.DataFrame):
    marginal_results = {}
    spear_results = {}
    unique_decays = data["weight_decay"].unique()
    for decay in unique_decays:
        subset = data[data["weight_decay"] == decay]
        id = "_".join(subset["unique_id"].str.split("_")[0][:2]) + f"_wd_{decay}"
        mean_marginal = np.mean(subset["marginal"])
        std_marginal = np.std(subset["marginal"])
        marginal_results[id] = (mean_marginal, std_marginal)

        mean_spear = np.mean(subset["spearman_rho"])
        std_spear = np.std(subset["spearman_rho"])
        spear_results[id] = (mean_spear, std_spear)

    return marginal_results, spear_results


gb1_ae_marginal_results, gb1_ae_spear_results = summary_data(gb1_ae)
gb1_a_marginal_results, gb1_a_spear_results = summary_data(gb1_a)
gb1_e_marginal_results, gb1_e_spear_results = summary_data(gb1_e)


# +


def plot(marginal_results: dict, spear_results: dict, title):
    sorted_marginal_keys = sorted(
        marginal_results.keys(), key=lambda x: float(x.split("_")[-1])
    )
    sorted_marginal_values = [marginal_results[k] for k in sorted_marginal_keys]

    # Sort the data for the second subplot
    sorted_spear_keys = sorted(
        spear_results.keys(), key=lambda x: float(x.split("_")[-1])
    )
    sorted_spear_values = [spear_results[k] for k in sorted_spear_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # Create bar plot for the first subplot
    bars = ax1.bar(
        sorted_marginal_keys,
        [x[0] for x in sorted_marginal_values],
        yerr=[x[1] for x in sorted_marginal_values],
    )
    ax1.set_xlabel("Weight decay")
    ax1.set_ylabel("log(P(X))")
    ax1.set_xticklabels(
        [float(x.split("_")[-1]) for x in sorted_marginal_keys], rotation=45
    )

    # Create bar plot for the second subplot
    bars = ax2.bar(
        sorted_spear_keys,
        [x[0] for x in sorted_spear_values],
        yerr=[x[1] for x in sorted_spear_values],
    )
    ax2.set_xlabel("Weight decay")
    ax2.set_ylabel("Spearman Rho")
    ax2.set_xticklabels(
        [float(x.split("_")[-1]) for x in sorted_spear_keys], rotation=45
    )

    # Set a title for the entire figure
    fig.suptitle(title)

    # Show plot
    plt.show()


plot(gb1_ae_marginal_results, gb1_ae_spear_results, "GB1 AE")
plot(gb1_a_marginal_results, gb1_a_spear_results, "GB1 A")
plot(gb1_e_marginal_results, gb1_e_spear_results, "GB1 E")

# -

# ### A4

# +
path = "/Users/sebs_mac/k_fold_results/a4/"

metric_paths = [
    "_".join(subdir.split("_")[:2]) + "_5_fold_" + "_".join(subdir.split("_")[2:])
    for subdir in os.listdir(path)
    if subdir != ".DS_Store"
]
metrics = [
    pd.read_csv(
        f"{path}{'_'.join(subdir.split('_')[:2] + subdir.split('_')[-2:])}/{subdir}_r1_metrics.csv"
    )
    for subdir in metric_paths
]
all_metrics = pd.concat(metrics)
dataset = [x[1] for x in all_metrics["unique_id"].str.split("_")]
decay = [x[5] for x in all_metrics["unique_id"].str.split("_")]
all_metrics["dataset"] = dataset
all_metrics["weight_decay"] = decay

a4_ae = all_metrics[all_metrics["dataset"] == "ae"]
a4_a = all_metrics[all_metrics["dataset"] == "a"]
a4_e = all_metrics[all_metrics["dataset"] == "e"]

# +

a4_ae_marginal_results, a4_ae_spear_results = summary_data(a4_ae)
a4_a_marginal_results, a4_a_spear_results = summary_data(a4_a)
a4_e_marginal_results, a4_e_spear_results = summary_data(a4_e)

plot(a4_ae_marginal_results, a4_ae_spear_results, "A4 AE")
plot(a4_a_marginal_results, a4_a_spear_results, "A4 A")
plot(a4_e_marginal_results, a4_e_spear_results, "A4 E")


# -

# ### GFP

# +
path = "/Users/sebs_mac/k_fold_results/gfp/"

metric_paths = [
    "_".join(subdir.split("_")[:2]) + "_5_fold_" + "_".join(subdir.split("_")[2:])
    for subdir in os.listdir(path)
    if subdir != ".DS_Store"
]
metrics = [
    pd.read_csv(
        f"{path}{'_'.join(subdir.split('_')[:2] + subdir.split('_')[-2:])}/{subdir}_r1_metrics.csv"
    )
    for subdir in metric_paths
]
all_metrics = pd.concat(metrics)
dataset = [x[1] for x in all_metrics["unique_id"].str.split("_")]
decay = [x[5] for x in all_metrics["unique_id"].str.split("_")]
all_metrics["dataset"] = dataset
all_metrics["weight_decay"] = decay

gfp_ae = all_metrics[all_metrics["dataset"] == "ae"]
gfp_a = all_metrics[all_metrics["dataset"] == "a"]
gfp_e = all_metrics[all_metrics["dataset"] == "e"]


gfp_ae_marginal_results, gfp_ae_spear_results = summary_data(gfp_ae)
gfp_a_marginal_results, gfp_a_spear_results = summary_data(gfp_a)
gfp_e_marginal_results, gfp_e_spear_results = summary_data(gfp_e)

plot(gfp_ae_marginal_results, gfp_ae_spear_results, "GFP AE")
plot(gfp_a_marginal_results, gfp_a_spear_results, "GFP A")
plot(gfp_e_marginal_results, gfp_e_spear_results, "GFP E")
# -

# ### MAFG

# +
path = "/Users/sebs_mac/k_fold_results/mafg/"

metric_paths = [
    "_".join(subdir.split("_")[:2]) + "_5_fold_" + "_".join(subdir.split("_")[2:])
    for subdir in os.listdir(path)
    if subdir != ".DS_Store"
]
metrics = [
    pd.read_csv(
        f"{path}{'_'.join(subdir.split('_')[:2] + subdir.split('_')[-2:])}/{subdir}_r1_metrics.csv"
    )
    for subdir in metric_paths
]
all_metrics = pd.concat(metrics)
dataset = [x[1] for x in all_metrics["unique_id"].str.split("_")]
decay = [x[5] for x in all_metrics["unique_id"].str.split("_")]
all_metrics["dataset"] = dataset
all_metrics["weight_decay"] = decay

mafg_ae = all_metrics[all_metrics["dataset"] == "ae"]
mafg_a = all_metrics[all_metrics["dataset"] == "a"]
mafg_e = all_metrics[all_metrics["dataset"] == "e"]
mafg_ae_marginal_results, mafg_ae_spear_results = summary_data(mafg_ae)
mafg_a_marginal_results, mafg_a_spear_results = summary_data(mafg_a)
mafg_e_marginal_results, mafg_e_spear_results = summary_data(mafg_e)

plot(mafg_ae_marginal_results, mafg_ae_spear_results, "MAFG AE")
plot(mafg_a_marginal_results, mafg_a_spear_results, "MAFG A")
plot(mafg_e_marginal_results, mafg_e_spear_results, "MAFG E")
# -

# ### GCN4

# +
path = "/Users/sebs_mac/k_fold_results/gcn4/"

metric_paths = [
    "_".join(subdir.split("_")[:2]) + "_5_fold_" + "_".join(subdir.split("_")[2:])
    for subdir in os.listdir(path)
    if subdir != ".DS_Store"
]
metrics = [
    pd.read_csv(
        f"{path}{'_'.join(subdir.split('_')[:2] + subdir.split('_')[-2:])}/{subdir}_r1_metrics.csv"
    )
    for subdir in metric_paths
]
all_metrics = pd.concat(metrics)
dataset = [x[1] for x in all_metrics["unique_id"].str.split("_")]
decay = [x[5] for x in all_metrics["unique_id"].str.split("_")]
all_metrics["dataset"] = dataset
all_metrics["weight_decay"] = decay

gcn4_ae = all_metrics[all_metrics["dataset"] == "ae"]
gcn4_a = all_metrics[all_metrics["dataset"] == "a"]
gcn4_e = all_metrics[all_metrics["dataset"] == "e"]
gcn4_ae_marginal_results, gcn4_ae_spear_results = summary_data(gcn4_ae)
gcn4_a_marginal_results, gcn4_a_spear_results = summary_data(gcn4_a)
gcn4_e_marginal_results, gcn4_e_spear_results = summary_data(gcn4_e)

plot(gcn4_ae_marginal_results, gcn4_ae_spear_results, "GCN4 AE")
plot(gcn4_a_marginal_results, gcn4_a_spear_results, "GCN4 A")
plot(gcn4_e_marginal_results, gcn4_e_spear_results, "GCN4 E")
