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

# # GB1 analysis
#

import pandas as pd
import matplotlib.pyplot as plt
import src.utils.seq_tools as st
import numpy as np
import logomaker as lm
import src.utils.statistics as stats
from matplotlib.patches import Patch
from scipy.stats import wasserstein_distance


# +
def plot_entropy(
    ancestors,
    extants,
    anc_count,
    ext_count,
    mutations,
    protein,
    title,
    ax,
    start=0,
    end=None,
    max_entropy=3,
):

    e_col_entropy = stats.calc_shannon_entropy(extants)
    a_col_entropy = stats.calc_shannon_entropy(ancestors)

    if end is not None:
        # across entire sequnece
        xticks = range(start, end)
        e_col_entropy = e_col_entropy[start:end]
        a_col_entropy = a_col_entropy[start:end]

    else:
        xticks = range(start, len(a_col_entropy))

    fig = plt.figure(figsize=(12, 8))
    ax.plot(xticks, a_col_entropy, alpha=1, color="orange")
    ax.plot(xticks, e_col_entropy, alpha=1, color="blue")
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("Entropy")
    ax.set_ylim(0, max_entropy)

    legend_elements = [
        Patch(
            facecolor="r", edgecolor="black", linestyle="--", label="mutagenesis sites"
        ),
        Patch(
            facecolor="orange",
            edgecolor="black",
            label=f"{protein} ancestors ({anc_count} seqs)",
        ),
        Patch(
            facecolor="blue",
            edgecolor="black",
            label=f"{protein} extants ({ext_count} seqs)",
        ),
    ]

    ax.legend(
        handles=legend_elements,
    )
    # if end is not None:
    #     ax.set_xticks(xticks, labels=xticks)

    for mutation in mutations:
        ax.axvline(x=mutation, color="r", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_title(f"{protein}: {title}")
    # plt.show()


def calc_column_wasserstein_distances(ancestors, extants, ancestors_extants, variants):
    """
    Returns:
    wd_a_to_e, wd_a_to_ae, wd_e_to_ae, wd_a_to_v, wd_e_to_v, wd_ae_to_v
    """

    if ancestors.split(".")[-1] in ["fasta", "aln"]:
        a = st.read_aln_file(ancestors)
    else:
        a = pd.read_pickle(ancestors)

    if ancestors_extants.split(".")[-1] in ["fasta", "aln"]:
        ae = st.read_aln_file(ancestors_extants)
    else:
        ae = pd.read_pickle(ancestors_extants)

    if extants.split(".")[-1] in ["fasta", "aln"]:
        e = st.read_aln_file(extants)
    else:
        e = pd.read_pickle(extants)

    var = pd.read_csv(variants)
    var = var.rename(columns={"mutated_sequence": "sequence", "mutant": "id"})

    a_ppm = stats.calc_position_prob_matrix(a)
    e_ppm = stats.calc_position_prob_matrix(e)
    ae_ppm = stats.calc_position_prob_matrix(ae)
    var_ppm = stats.calc_position_prob_matrix(var)

    wd_a_to_e = []
    wd_a_to_ae = []
    wd_e_to_ae = []
    wd_a_to_v = []
    wd_e_to_v = []
    wd_ae_to_v = []

    for i in range(a_ppm.shape[1]):
        col_a_i = a_ppm[:, i]
        col_e_i = e_ppm[:, i]
        col_ae_i = ae_ppm[:, i]
        col_v_i = var_ppm[:, i]

        wd_a_to_e.append(
            wasserstein_distance(
                np.arange(len(col_a_i)), np.arange(len(col_a_i)), col_a_i, col_e_i
            )
        )
        wd_a_to_ae.append(
            wasserstein_distance(
                np.arange(len(col_a_i)), np.arange(len(col_ae_i)), col_a_i, col_ae_i
            )
        )
        wd_e_to_ae.append(
            wasserstein_distance(
                np.arange(len(col_a_i)), np.arange(len(col_ae_i)), col_e_i, col_ae_i
            )
        )

        wd_a_to_v.append(
            wasserstein_distance(
                np.arange(len(col_a_i)), np.arange(len(col_a_i)), col_a_i, col_v_i
            )
        )
        wd_e_to_v.append(
            wasserstein_distance(
                np.arange(len(col_a_i)), np.arange(len(col_ae_i)), col_e_i, col_v_i
            )
        )
        wd_ae_to_v.append(
            wasserstein_distance(
                np.arange(len(col_a_i)), np.arange(len(col_ae_i)), col_ae_i, col_v_i
            )
        )

    return wd_a_to_e, wd_a_to_ae, wd_e_to_ae, wd_a_to_v, wd_e_to_v, wd_ae_to_v


# -

meta = pd.read_csv("/Users/sebs_mac/git_repos/dms_data/DMS_substitutions.csv")
# print(meta.columns)
meta[meta["DMS_id"] == "SPG1_STRSG_Wu_2016"]["target_seq"].values[0]

# The following residues were modified
#
# They all lie within 15 residues of each other
#
# The distribution of data is very skewed though
# "While most mutants had a lower fitness compared to WT (fitness < 1), 2.4% of mutants were beneficial (fitness > 1)."

# ### GB1 - variant distribution

gb1_variants = pd.read_csv(
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/SPG1_STRSG_Wu_2016.csv"
)
gb1_variants.head()

# +
gb1_variants["DMS_score"].describe()

plt.hist(gb1_variants["DMS_score"], bins=40)
plt.ylabel("Number of variants")
plt.xlabel("Fitness")
plt.title("GB1 variants")
plt.show()
# -

# ### GB1 - entropy

# +
################ UNCURATED ##############

aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/alns/"
gb1_e = pd.read_pickle(aln_path + "gb1_extants_no_dupes.pkl")
gb1_a = pd.read_pickle(aln_path + "gb1_ancestors_no_dupes.pkl")


################### CURATED ANCESTORS & EXTANTS ##################
aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/profile_creation/alns/"
curated_gb1_e = st.read_aln_file(aln_path + "gb1_0.32_sim_extants_no_dupes.aln")
curated_gb1_a = st.read_aln_file(aln_path + "gb1_0.32_sim_ancestors_no_dupes.aln")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

plot_entropy(
    ancestors=gb1_a,
    extants=gb1_e,
    anc_count=12733,
    ext_count=2286,
    mutations=[264, 265, 266, 279],
    protein="GB1",
    title="Column entropy ancestors vs extants",
    ax=ax1,
)

plot_entropy(
    ancestors=curated_gb1_a,
    extants=curated_gb1_e,
    anc_count=8343,
    ext_count=2286,
    mutations=[264, 265, 266, 279],
    protein="Curated GB1",
    title="Column entropy curated ancestors vs curated extants",
    ax=ax2,
)

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

# do across a smaller range
plot_entropy(
    ancestors=gb1_a,
    extants=gb1_e,
    anc_count=12733,
    ext_count=2286,
    mutations=[264, 265, 266, 279],
    protein="GB1",
    title="Column entropy ancestors vs extants",
    start=260,
    end=282,
    ax=ax1,
)


plot_entropy(
    ancestors=curated_gb1_a,
    extants=curated_gb1_e,
    anc_count=8343,
    ext_count=2286,
    mutations=[264, 265, 266, 279],
    protein="Curated GB1",
    title="Column entropy curated ancestors vs curated extants",
    start=260,
    end=282,
    ax=ax2,
)

plt.show()

# +

aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/profile_creation/alns/"
curated_gb1_e = st.read_aln_file(aln_path + "gb1_0.32_sim_extants_no_dupes.aln")
curated_gb1_a = st.read_aln_file(aln_path + "gb1_0.32_sim_ancestors_no_dupes.aln")

var = pd.read_csv(
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/SPG1_STRSG_Wu_2016.csv"
)
var.rename(columns={"mutant": "id", "mutated_sequence": "sequence"}, inplace=True)
print(var.columns)

fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
plot_entropy(
    ancestors=curated_gb1_e,
    extants=var,
    anc_count=12733,
    ext_count=2286,
    mutations=[264, 265, 266, 279],
    protein="GB1",
    title="extants entropy vs variants",
    start=260,
    end=282,
    ax=ax1,
)

plt.show()

fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
plot_entropy(
    ancestors=curated_gb1_a,
    extants=var,
    anc_count=12733,
    ext_count=2286,
    mutations=[264, 265, 266, 279],
    protein="GB1",
    title="ancestors entropy vs variants",
    start=260,
    end=282,
    ax=ax1,
)

plt.show()


aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/alns/"
gb1_a = pd.read_pickle(aln_path + "gb1_ancestors_no_dupes.pkl")

fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
plot_entropy(
    ancestors=curated_gb1_a,
    extants=gb1_a,
    anc_count=12733,
    ext_count=2286,
    mutations=[264, 265, 266, 279],
    protein="GB1",
    title="ancestors entropy vs variants",
    start=260,
    end=282,
    ax=ax1,
)

plt.show()
# -


# #### GB1 - Wasserstein distance

# +


################ UNCURATED ##############
path = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/alns/"
wd_a_to_e, wd_a_to_ae, wd_e_to_ae, wd_a_to_v, wd_e_to_v, wd_ae_to_v = (
    calc_column_wasserstein_distances(
        ancestors=path + "gb1_ancestors_no_dupes.pkl",
        extants=path + "gb1_extants_no_dupes.pkl",
        ancestors_extants=path + "gb1_ancestors_extants_no_dupes.pkl",
        variants="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/SPG1_STRSG_Wu_2016.csv",
    )
)

print(f"Dist A-E: {np.mean(wd_a_to_e)}")
print(f"Dist A-AE: {np.mean(wd_a_to_ae)}")
print(f"Dist E-AE: {np.mean(wd_e_to_ae)}")

print(f"Dist A-V: {np.mean(wd_a_to_v)}")
print(f"Dist E-V: {np.mean(wd_e_to_v)}")
print(f"Dist AE-V: {np.mean(wd_ae_to_v)}")

################### CURATED ANCESTORS & EXTANTS ##################

path = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/profile_creation/alns/"
(
    wd_curated_a_to_curated_e,
    wd_curated_a_to_curated_ae,
    wd_curated_e_to_curated_ae,
    wd_curated_a_to_v,
    wd_curated_e_to_v,
    wd_curated_ae_to_v,
) = calc_column_wasserstein_distances(
    ancestors=path + "gb1_0.32_sim_ancestors_no_dupes.aln",
    extants=path + "gb1_0.32_sim_extants_no_dupes.aln",
    ancestors_extants=path + "gb1_0.32_sim_ancestors_0.32_sim_extants_no_dupes.aln",
    variants="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/SPG1_STRSG_Wu_2016.csv",
)

print(f"Dist curated A-E: {np.mean(wd_curated_a_to_curated_e)}")
print(f"Dist curated A-AE: {np.mean(wd_curated_a_to_curated_ae)}")
print(f"Dist curated E-AE: {np.mean(wd_curated_e_to_curated_ae)}")

print(f"Dist curated A-V: {np.mean(wd_curated_a_to_v)}")
print(f"Dist curated E-V: {np.mean(wd_curated_e_to_v)}")
print(f"Dist curated AE-V: {np.mean(wd_curated_ae_to_v)}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.bar(
    np.arange(6),
    [
        np.mean(wd_a_to_e),
        np.mean(wd_a_to_ae),
        np.mean(wd_e_to_ae),
        np.mean(wd_a_to_v),
        np.mean(wd_e_to_v),
        np.mean(wd_ae_to_v),
    ],
)
ax1.set_title("GB1: Uncurated Dataset comparisons")
ax1.set_ylabel("Wasserstein distance")
ax1.set_xticks(
    ticks=np.arange(6),
    labels=["Anc-Ex", "Anc-Anc/Ex", "E-Anc/Ex", "Anc-Var", "Ex-Var", "Anc/Ex-Var"],
    rotation=45,
)
ax1.set_ylim(0, 6)
ax1.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)


ax2.bar(
    np.arange(6),
    [
        np.mean(wd_curated_a_to_curated_e),
        np.mean(wd_curated_a_to_curated_ae),
        np.mean(wd_curated_e_to_curated_ae),
        np.mean(wd_curated_a_to_v),
        np.mean(wd_curated_e_to_v),
        np.mean(wd_curated_ae_to_v),
    ],
)
ax2.set_title("GB1: Curated Dataset comparisons")
ax2.set_ylabel("Wasserstein distance")
ax2.set_xticks(
    ticks=np.arange(6),
    labels=["Anc-Ex", "Anc-Anc/Ex", "E-Anc/Ex", "Anc-Var", "Ex-Var", "Anc/Ex-Var"],
    rotation=45,
)
ax2.set_ylim(0, 6)
ax2.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)


plt.show()


# +
# wd_a_to_v, wd_curated_e_to_v, wd_ae_to_v


def plot_wasserstein_dist(data, mutations, title, ax, ylim=10):
    scores = [x for idx, x in enumerate(data) if idx in mutations]
    ax.bar(np.arange(len(mutations)), scores)
    ax.set_xticks(ticks=np.arange(len(mutations)), labels=mutations)
    ax.set_title(title)
    ax.set_ylim(0, ylim)
    ax.set_xlabel("Mutated position index")
    ax.set_ylabel("Wasserstein distance")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
mutations = [264, 265, 266, 279]
plot_wasserstein_dist(
    wd_a_to_v, mutations=mutations, title="Uncurated ancestors to variants", ax=ax1
)


mutations = [264, 265, 266, 279]
plot_wasserstein_dist(
    wd_curated_a_to_v,
    mutations=mutations,
    title="Curated ancestors to variants",
    ax=ax2,
)

plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
mutations = [264, 265, 266, 279]
plot_wasserstein_dist(
    wd_e_to_v, mutations=mutations, title="Uncurated extants to variants", ax=ax1
)


mutations = [264, 265, 266, 279]
plot_wasserstein_dist(
    wd_curated_e_to_v, mutations=mutations, title="Curated extants to variants", ax=ax2
)

plt.show()

# +
################  ##############
xticks = range(0, len(wd_a_to_e))

fig = plt.figure(figsize=(12, 8))
plt.plot(xticks, wd_a_to_e, alpha=1, color="blue", label="Ancestors-Extants")
plt.plot(
    xticks,
    wd_curated_a_to_curated_e,
    alpha=1,
    color="orange",
    label="Curated Ancestors- Curated Extants",
)
mutations = [264, 265, 266, 279]
for mutation in mutations:
    plt.axvline(x=mutation, color="r", linestyle="--", linewidth=1, alpha=1)

legend_elements = [
    Patch(facecolor="r", edgecolor="black", linestyle="--", label="mutagenesis sites"),
    Patch(
        facecolor="blue",
        edgecolor="black",
        linestyle="--",
        label="Ancestors to Extants ",
    ),
    Patch(
        facecolor="orange",
        edgecolor="black",
        linestyle="--",
        label="Curated ancestors to Curated Extants",
    ),
]

plt.legend(
    handles=legend_elements,
)
plt.ylabel("Wasserstein distance")
plt.xlabel("Sequence index")
plt.title("Wasserstein distance per column between Ancestors and Extants (GB1)")
plt.show()


################### ONLY ACROSS MUTATIONS ##################
xticks = range(0, len(wd_curated_a_to_curated_e))

fig = plt.figure(figsize=(12, 8))
plt.plot(
    xticks[260:282],
    wd_a_to_e[260:282],
    alpha=1,
    color="blue",
    label="Ancestors-Extants",
)
plt.plot(
    xticks[260:282],
    wd_curated_a_to_curated_e[260:282],
    alpha=1,
    color="orange",
    label="Curated ancestors-Extants",
)


mutations = [264, 265, 266, 279]
for mutation in mutations:
    plt.axvline(x=mutation, color="r", linestyle="--", linewidth=1, alpha=1)

legend_elements = [
    Patch(facecolor="r", edgecolor="black", linestyle="--", label="mutagenesis sites"),
    Patch(
        facecolor="blue",
        edgecolor="black",
        linestyle="--",
        label="Ancestors to Extants ",
    ),
    Patch(
        facecolor="orange",
        edgecolor="black",
        linestyle="--",
        label="Curated ancestors to Curated Extants",
    ),
]

plt.legend(
    handles=legend_elements,
)
plt.ylabel("Wasserstein distance")
plt.xlabel("Sequence index")

plt.title("Wasserstein distance per column between Ancestors and Extants (GB1)")
plt.show()

################### ONLY ACROSS MUTATIONS ##################
xticks = range(0, len(wd_curated_a_to_curated_e))

fig = plt.figure(figsize=(12, 8))
plt.plot(xticks[260:282], wd_a_to_v[260:282], alpha=1, color="blue")
plt.plot(xticks[260:282], wd_curated_a_to_v[260:282], alpha=1, color="orange")

plt.plot(xticks[260:282], wd_e_to_v[260:282], alpha=1, color="purple")
plt.plot(xticks[260:282], wd_curated_e_to_v[260:282], alpha=1, color="green")

mutations = [264, 265, 266, 279]
for mutation in mutations:
    plt.axvline(x=mutation, color="r", linestyle="--", linewidth=1, alpha=1)

legend_elements = [
    Patch(facecolor="r", edgecolor="black", linestyle="--", label="mutagenesis sites"),
    Patch(
        facecolor="blue",
        edgecolor="black",
        linestyle="--",
        label="Ancestors to Variants ",
    ),
    Patch(
        facecolor="orange",
        edgecolor="black",
        linestyle="--",
        label="Curated ancestors to Variants",
    ),
    Patch(
        facecolor="purple",
        edgecolor="black",
        linestyle="--",
        label="Extants to Variants ",
    ),
    Patch(
        facecolor="green",
        edgecolor="black",
        linestyle="--",
        label="Curated extants to Variants",
    ),
]

plt.legend(
    handles=legend_elements,
)
plt.ylabel("Wasserstein distance")
plt.xlabel("Sequence index")

plt.title("Wasserstein distance per column between Ancestors and Variants (GB1)")
plt.show()
# -

# ### GB1 - curated sequence weights distribtutions

# +


aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/alns/"
gb1_e = pd.read_pickle(aln_path + "gb1_extants_no_dupes.pkl")
gb1_a = pd.read_pickle(aln_path + "gb1_ancestors_no_dupes.pkl")
gb1_ae = pd.read_pickle(aln_path + "gb1_ancestors_extants_no_dupes.pkl")

aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/profile_creation/alns/"
curated_gb1_e = st.read_aln_file(aln_path + "gb1_0.32_sim_extants_no_dupes.aln")
curated_gb1_a = st.read_aln_file(aln_path + "gb1_0.32_sim_ancestors_no_dupes.aln")
curated_gb1_ae = st.read_aln_file(
    aln_path + "gb1_0.32_sim_ancestors_0.32_sim_extants_no_dupes.aln"
)

datasets = [gb1_e, gb1_a, gb1_ae, curated_gb1_e, curated_gb1_a, curated_gb1_ae]

for data in datasets:
    numpy_aln, _, _ = st.convert_msa_numpy_array(data)
    data["seq_sim"] = st.reweight_by_seq_similarity(numpy_aln, 0.2)
    data["col_sim"] = st.position_based_seq_weighting(numpy_aln, 4)


# +


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.hist(np.log(gb1_a["col_sim"]), color="red", alpha=1, bins=20, label="Ancestors")
ax1.hist(
    np.log(curated_gb1_a["col_sim"]),
    color="blue",
    alpha=0.5,
    bins=20,
    label="Curated Ancestors",
)

ax1.set_ylabel("Count")
ax1.set_xlabel("ln(Sequence weight)")

ax1.legend()


ax2.hist(np.log(gb1_e["col_sim"]), color="red", alpha=1, bins=20, label="Extants")
ax2.hist(
    np.log(curated_gb1_e["col_sim"]),
    color="blue",
    alpha=0.5,
    bins=20,
    label="Curated extants",
)

ax2.set_ylabel("Count")
ax2.set_xlabel("ln(Sequence weight)")

ax2.set_title("Curated GB1 Sequence Weight Distributions")
ax2.legend()


ax3.hist(
    np.log(gb1_ae["col_sim"]), color="red", alpha=1, bins=20, label="Ancestors/Extants"
)
ax3.hist(
    np.log(curated_gb1_ae["col_sim"]),
    color="blue",
    alpha=0.5,
    bins=20,
    label="Curated Ancestors/Extants",
)
ax3.set_ylabel("Count")
ax3.set_xlabel("ln(Sequence weight)")

ax3.legend()

plt.show()
# -


# # GFP

gfp_variants = pd.read_csv(
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv"
)
gfp_variants.shape

plt.hist(gfp_variants["DMS_score"], bins=40)
plt.ylabel("Number of variants")
plt.xlabel("Fitness")
plt.title("GFP variants")
plt.show()

# +
aln_path = (
    "/Users/sebs_mac/uni_OneDrive/honours/data/gfp/independent_runs/no_synthetic/alns/"
)
gfp_e = pd.read_pickle(aln_path + "gfp_extants_no_syn_no_dupes.pkl")
gfp_a = pd.read_pickle(aln_path + "gfp_ancestors_no_syn_no_dupes.pkl")

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))

plot_entropy(
    ancestors=gfp_a,
    extants=gfp_e,
    anc_count=497,
    ext_count=324,
    mutations=[3, 237],
    protein="GFP",
    title="Column entropy ancestors vs extants",
    ax=ax1,
)


plt.show()
# -


# ##### GFP - Wasserstein distance

# +
path = (
    "/Users/sebs_mac/uni_OneDrive/honours/data/gfp/independent_runs/no_synthetic/alns/"
)
wd_a_to_e, wd_a_to_ae, wd_e_to_ae, wd_a_to_v, wd_e_to_v, wd_ae_to_v = (
    calc_column_wasserstein_distances(
        ancestors=path + "gfp_ancestors_no_syn_no_dupes.pkl",
        extants=path + "gfp_extants_no_syn_no_dupes.pkl",
        ancestors_extants=path + "gfp_ancestors_extants_no_syn_no_dupes.pkl",
        variants="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv",
    )
)

print(f"Dist A-E: {np.mean(wd_a_to_e)}")
print(f"Dist A-AE: {np.mean(wd_a_to_ae)}")
print(f"Dist E-AE: {np.mean(wd_e_to_ae)}")

print(f"Dist A-V: {np.mean(wd_a_to_v)}")
print(f"Dist E-V: {np.mean(wd_e_to_v)}")
print(f"Dist AE-V: {np.mean(wd_ae_to_v)}")

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 6))

ax1.bar(
    np.arange(6),
    [
        np.mean(wd_a_to_e),
        np.mean(wd_a_to_ae),
        np.mean(wd_e_to_ae),
        np.mean(wd_a_to_v),
        np.mean(wd_e_to_v),
        np.mean(wd_ae_to_v),
    ],
)
ax1.set_title("GFP: Uncurated Dataset comparisons")
ax1.set_ylabel("Wasserstein distance")
ax1.set_xticks(
    ticks=np.arange(6),
    labels=["Anc-Ex", "Anc-Anc/Ex", "E-Anc/Ex", "Anc-Var", "Ex-Var", "Anc/Ex-Var"],
    rotation=45,
)
ax1.set_ylim(0, 1.5)
ax1.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
plt.show()


# +

################ UNCURATED ##############
xticks = range(0, len(wd_a_to_e))

fig = plt.figure(figsize=(12, 8))
plt.plot(xticks, wd_a_to_e, alpha=1, color="blue", label="Ancestors-Extants")
mutations = [3, 237]
for mutation in mutations:
    plt.axvline(x=mutation, color="r", linestyle="--", linewidth=1, alpha=1)

legend_elements = [
    Patch(facecolor="r", edgecolor="black", linestyle="--", label="mutagenesis sites"),
    Patch(
        facecolor="blue",
        edgecolor="black",
        linestyle="--",
        label="Ancestors to Extants ",
    ),
]

plt.legend(
    handles=legend_elements,
)
plt.ylabel("Wasserstein distance")
plt.xlabel("Sequence index")
plt.title("Wasserstein distance per column between Ancestors and Extants (GFP)")
plt.show()
# -

# # A4

# +
# 672-713 (673) /Users/sebs_mac/uni_OneDrive/honours/data/a4_human/alns A4_HUMAN_Seuma_2022

a4_variants = pd.read_csv(
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/A4_HUMAN_Seuma_2022.csv"
)
a4_variants.head()

# +
# all the mutation positoins will be here
a4_muts = set()

for x in a4_variants["mutant"]:
    data = x.split(":")
    a = data[0][1:-1]
    a4_muts.add(a)
    if len(data) == 2:
        b = data[1][1:-1]
        a4_muts.add(b)
len(a4_muts)
print(min(a4_muts), max(a4_muts))

# +
a4_variants["DMS_score"].describe()

plt.hist(a4_variants["DMS_score"], bins=40)
plt.ylabel("Number of variants")
plt.xlabel("Fitness")
plt.title("A4 variants")
plt.show()

# +
aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/a4_human/alns/"
a4_e = pd.read_pickle(aln_path + "a4_extants_no_dupes.pkl")
a4_a = pd.read_pickle(aln_path + "a4_ancestors_no_dupes.pkl")


################### CURATED ANCESTORS & EXTANTS ##################
aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/a4_human/curated/alns/"
curated_a4_e = st.read_aln_file(aln_path + "a4_0.6_sim_extants_no_dupes.aln")
curated_a4_a = st.read_aln_file(aln_path + "a4_0.6_sim_ancestors_no_dupes.aln")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

plot_entropy(
    ancestors=a4_a,
    extants=a4_e,
    anc_count=29940,
    ext_count=4936,
    mutations=[671, 712],
    protein="A4",
    title="Column entropy ancestors vs extants",
    ax=ax1,
    max_entropy=2.5,
)


plot_entropy(
    ancestors=curated_a4_a,
    extants=curated_a4_e,
    anc_count=9755,
    ext_count=1978,
    mutations=[671, 712],
    protein="A4",
    title="Column entropy curated ancestors vs curated extants",
    ax=ax2,
    max_entropy=2.5,
)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

plot_entropy(
    ancestors=a4_a,
    extants=a4_e,
    anc_count=29940,
    ext_count=4936,
    mutations=[671, 712],
    protein="A4",
    title="Column entropy ancestors vs extants",
    ax=ax1,
    max_entropy=2.5,
    start=670,
    end=714,
)


plot_entropy(
    ancestors=curated_a4_a,
    extants=curated_a4_e,
    anc_count=9755,
    ext_count=1978,
    mutations=[671, 712],
    protein="A4",
    title="Column entropy curated ancestors vs curated extants",
    ax=ax2,
    max_entropy=2.5,
    start=670,
    end=714,
)
plt.show()
# -


# ### A4 Wasserstein distance

# +
path = "/Users/sebs_mac/uni_OneDrive/honours/data/a4_human/alns/"

wd_a_to_e, wd_a_to_ae, wd_e_to_ae, wd_a_to_v, wd_e_to_v, wd_ae_to_v = (
    calc_column_wasserstein_distances(
        ancestors=path + "a4_ancestors_no_dupes.pkl",
        extants=path + "a4_extants_no_dupes.pkl",
        ancestors_extants=path + "a4_ancestors_extants_no_dupes.pkl",
        variants="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/A4_HUMAN_Seuma_2022.csv",
    )
)

print(f"Dist A-E: {np.mean(wd_a_to_e)}")
print(f"Dist A-AE: {np.mean(wd_a_to_ae)}")
print(f"Dist E-AE: {np.mean(wd_e_to_ae)}")

print(f"Dist A-V: {np.mean(wd_a_to_v)}")
print(f"Dist E-V: {np.mean(wd_e_to_v)}")
print(f"Dist AE-V: {np.mean(wd_ae_to_v)}")

################### CURATED ANCESTORS & EXTANTS ##################

path = "/Users/sebs_mac/uni_OneDrive/honours/data/a4_human/curated/alns/"
(
    wd_curated_a_to_curated_e,
    wd_curated_a_to_curated_ae,
    wd_curated_e_to_curated_ae,
    wd_curated_a_to_v,
    wd_curated_e_to_v,
    wd_curated_ae_to_v,
) = calc_column_wasserstein_distances(
    ancestors=path + "a4_0.6_sim_ancestors_no_dupes.aln",
    extants=path + "a4_0.6_sim_extants_no_dupes.aln",
    ancestors_extants=path + "a4_0.6_sim_ancestors_0.6_sim_extants_no_dupes.aln",
    variants="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/A4_HUMAN_Seuma_2022.csv",
)

print(f"Dist curated A-E: {np.mean(wd_curated_a_to_curated_e)}")
print(f"Dist curated A-AE: {np.mean(wd_curated_a_to_curated_ae)}")
print(f"Dist curated E-AE: {np.mean(wd_curated_e_to_curated_ae)}")

print(f"Dist curated A-V: {np.mean(wd_curated_a_to_v)}")
print(f"Dist curated E-V: {np.mean(wd_curated_e_to_v)}")
print(f"Dist curated AE-V: {np.mean(wd_curated_ae_to_v)}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.bar(
    np.arange(6),
    [
        np.mean(wd_a_to_e),
        np.mean(wd_a_to_ae),
        np.mean(wd_e_to_ae),
        np.mean(wd_a_to_v),
        np.mean(wd_e_to_v),
        np.mean(wd_ae_to_v),
    ],
)
ax1.set_title("A4: Uncurated Dataset comparisons")
ax1.set_ylabel("Wasserstein distance")
ax1.set_xticks(
    ticks=np.arange(6),
    labels=["Anc-Ex", "Anc-Anc/Ex", "E-Anc/Ex", "Anc-Var", "Ex-Var", "Anc/Ex-Var"],
    rotation=45,
)
ax1.set_ylim(0, 3)
ax1.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)


ax2.bar(
    np.arange(6),
    [
        np.mean(wd_curated_a_to_curated_e),
        np.mean(wd_curated_a_to_curated_ae),
        np.mean(wd_curated_e_to_curated_ae),
        np.mean(wd_curated_a_to_v),
        np.mean(wd_curated_e_to_v),
        np.mean(wd_curated_ae_to_v),
    ],
)
ax2.set_title("A4: Curated Dataset comparisons")
ax2.set_ylabel("Wasserstein distance")
ax2.set_xticks(
    ticks=np.arange(6),
    labels=["Anc-Ex", "Anc-Anc/Ex", "E-Anc/Ex", "Anc-Var", "Ex-Var", "Anc/Ex-Var"],
    rotation=45,
)
ax2.set_ylim(0, 3)
ax2.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)

plt.show()

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
mutations = [x for x in range(670, 714)]
plot_wasserstein_dist(
    wd_a_to_v, mutations=mutations, title="Uncurated ancestors to variants", ax=ax1
)

mutations = [x for x in range(670, 714)]
plot_wasserstein_dist(
    wd_curated_a_to_v,
    mutations=mutations,
    title="Curated ancestors to variants",
    ax=ax2,
)

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
mutations = [x for x in range(670, 714)]
plot_wasserstein_dist(
    wd_e_to_v, mutations=mutations, title="Uncurated extants to variants", ax=ax1
)

mutations = [x for x in range(670, 714)]
plot_wasserstein_dist(
    wd_curated_e_to_v, mutations=mutations, title="Curated extants to variants", ax=ax2
)

plt.show()


# +
xticks = range(0, len(wd_a_to_e))

fig = plt.figure(figsize=(12, 8))
plt.plot(xticks, wd_a_to_e, alpha=1, color="blue")
plt.plot(xticks, wd_curated_a_to_curated_e, alpha=1, color="orange")
mutations = [671, 712]
for mutation in mutations:
    plt.axvline(x=mutation, color="r", linestyle="--", linewidth=1, alpha=1)

legend_elements = [
    Patch(facecolor="r", edgecolor="black", linestyle="--", label="mutagenesis sites"),
    Patch(
        facecolor="blue",
        edgecolor="black",
        linestyle="--",
        label="Ancestors to Extants ",
    ),
    Patch(
        facecolor="orange",
        edgecolor="black",
        linestyle="--",
        label="Curated ancestors to Curated Extants",
    ),
]

plt.legend(
    handles=legend_elements,
)
plt.ylabel("Wasserstein distance")
plt.xlabel("Sequence index")
plt.title("Wasserstein distance per column between Ancestors and Extants (A4)")
plt.show()


################### ONLY ACROSS MUTATIONS ##################

xticks = range(0, len(wd_a_to_e))

fig = plt.figure(figsize=(12, 8))
plt.plot(xticks[670:714], wd_a_to_e[670:714], alpha=1, color="blue")
plt.plot(xticks[670:714], wd_curated_a_to_curated_e[670:714], alpha=1, color="orange")
mutations = [671, 712]
for mutation in mutations:
    plt.axvline(x=mutation, color="r", linestyle="--", linewidth=1, alpha=1)

legend_elements = [
    Patch(facecolor="r", edgecolor="black", linestyle="--", label="mutagenesis sites"),
    Patch(
        facecolor="blue",
        edgecolor="black",
        linestyle="--",
        label="Ancestors to Extants ",
    ),
    Patch(
        facecolor="orange",
        edgecolor="black",
        linestyle="--",
        label="Curated ancestors to Curated Extants",
    ),
]

plt.legend(
    handles=legend_elements,
)
plt.ylabel("Wasserstein distance")
plt.xlabel("Sequence index")
plt.title("Wasserstein distance per column between Ancestors and Extants (A4)")
plt.show()
# -


# # GCN4

gcn4_variants = pd.read_csv(
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GCN4_YEAST_Staller_2018.csv"
)
gcn4_variants.head()

# +
gcn4_variants["DMS_score"].describe()

plt.hist(gcn4_variants["DMS_score"], bins=40)
plt.ylabel("Number of variants")
plt.xlabel("Fitness")
plt.title("GCN4 variants")
plt.show()
# -

# ### GCN4 - entropy

# +
################ UNCURATED ##############

aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/gcn4/alns/"
gcn4_e = pd.read_pickle(aln_path + "gcn4_extants_no_dupes.pkl")
gcn4_a = pd.read_pickle(aln_path + "gcn4_ancestors_no_dupes.pkl")

################### CURATED ANCESTORS & EXTANTS ##################
aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/gcn4/curated/alns/"
curated_gcn4_e = st.read_aln_file(aln_path + "gcn4_0.3_sim_extants_no_dupes.aln")
curated_gcn4_a = st.read_aln_file(aln_path + "gcn4_0.3_sim_ancestors_no_dupes.aln")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

plot_entropy(
    ancestors=gcn4_a,
    extants=gcn4_e,
    anc_count=11623,
    ext_count=331,
    mutations=[101, 144],
    protein="GCN4",
    title="Column entropy ancestors vs extants",
    ax=ax1,
)

plot_entropy(
    ancestors=curated_gcn4_a,
    extants=curated_gcn4_e,
    anc_count=9171,
    ext_count=283,
    mutations=[101, 144],
    protein="Curated GCN4",
    title="Column entropy curated ancestors vs curated extants",
    ax=ax2,
)

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

# do across a smaller range
plot_entropy(
    ancestors=gcn4_a,
    extants=gcn4_e,
    anc_count=11623,
    ext_count=331,
    mutations=[101, 144],
    protein="GCN4",
    title="Column entropy ancestors vs extants",
    start=99,
    end=147,
    ax=ax1,
)


plot_entropy(
    ancestors=curated_gcn4_a,
    extants=curated_gcn4_e,
    anc_count=9171,
    ext_count=283,
    mutations=[101, 144],
    protein="Curated GCN4",
    title="Column entropy curated ancestors vs curated extants",
    start=99,
    end=147,
    ax=ax2,
)

plt.show()
# -


# ##### GCN4 - Wasserstein distance

# +


################ UNCURATED ##############
path = "/Users/sebs_mac/uni_OneDrive/honours/data/gcn4/alns/"
wd_a_to_e, wd_a_to_ae, wd_e_to_ae, wd_a_to_v, wd_e_to_v, wd_ae_to_v = (
    calc_column_wasserstein_distances(
        ancestors=path + "gcn4_ancestors_no_dupes.pkl",
        extants=path + "gcn4_extants_no_dupes.pkl",
        ancestors_extants=path + "gcn4_ancestors_extants_no_dupes.pkl",
        variants="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GCN4_YEAST_Staller_2018.csv",
    )
)

print(f"Dist A-E: {np.mean(wd_a_to_e)}")
print(f"Dist A-AE: {np.mean(wd_a_to_ae)}")
print(f"Dist E-AE: {np.mean(wd_e_to_ae)}")

print(f"Dist A-V: {np.mean(wd_a_to_v)}")
print(f"Dist E-V: {np.mean(wd_e_to_v)}")
print(f"Dist AE-V: {np.mean(wd_ae_to_v)}")


################### CURATED ANCESTORS & EXTANTS ##################

path = "/Users/sebs_mac/uni_OneDrive/honours/data/gcn4/curated/alns/"
(
    wd_curated_a_to_curated_e,
    wd_curated_a_to_curated_ae,
    wd_curated_e_to_curated_ae,
    wd_curated_a_to_v,
    wd_curated_e_to_v,
    wd_curated_ae_to_v,
) = calc_column_wasserstein_distances(
    ancestors=path + "gcn4_0.3_sim_ancestors_no_dupes.aln",
    extants=path + "gcn4_0.3_sim_extants_no_dupes.aln",
    ancestors_extants=path + "gcn4_0.3_sim_ancestors_0.3_sim_extants_no_dupes.aln",
    variants="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GCN4_YEAST_Staller_2018.csv",
)

print(f"Dist curated A-E: {np.mean(wd_curated_a_to_curated_e)}")
print(f"Dist curated A-AE: {np.mean(wd_curated_a_to_curated_ae)}")
print(f"Dist curated E-AE: {np.mean(wd_curated_e_to_curated_ae)}")

print(f"Dist curated A-V: {np.mean(wd_curated_a_to_v)}")
print(f"Dist curated E-V: {np.mean(wd_curated_e_to_v)}")
print(f"Dist curated AE-V: {np.mean(wd_curated_ae_to_v)}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.bar(
    np.arange(6),
    [
        np.mean(wd_a_to_e),
        np.mean(wd_a_to_ae),
        np.mean(wd_e_to_ae),
        np.mean(wd_a_to_v),
        np.mean(wd_e_to_v),
        np.mean(wd_ae_to_v),
    ],
)
ax1.set_title("GCN4: Uncurated Dataset comparisons")
ax1.set_ylabel("Wasserstein distance")
ax1.set_xticks(
    ticks=np.arange(6),
    labels=["Anc-Ex", "Anc-Anc/Ex", "E-Anc/Ex", "Anc-Var", "Ex-Var", "Anc/Ex-Var"],
    rotation=45,
)
ax1.set_ylim(0, 5)
ax1.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)


ax2.bar(
    np.arange(6),
    [
        np.mean(wd_curated_a_to_curated_e),
        np.mean(wd_curated_a_to_curated_ae),
        np.mean(wd_curated_e_to_curated_ae),
        np.mean(wd_curated_a_to_v),
        np.mean(wd_curated_e_to_v),
        np.mean(wd_curated_ae_to_v),
    ],
)
ax2.set_title("GCN4: Curated Dataset comparisons")
ax2.set_ylabel("Wasserstein distance")
ax2.set_xticks(
    ticks=np.arange(6),
    labels=["Anc-Ex", "Anc-Anc/Ex", "E-Anc/Ex", "Anc-Var", "Ex-Var", "Anc/Ex-Var"],
    rotation=45,
)
ax2.set_ylim(0, 5)
ax2.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)


plt.show()


# +
################ UNCURATED ##############
xticks = range(0, len(wd_a_to_e))

fig = plt.figure(figsize=(12, 8))
plt.plot(xticks, wd_a_to_e, alpha=1, color="blue", label="Ancestors-Extants")
plt.plot(
    xticks,
    wd_curated_a_to_curated_e,
    alpha=1,
    color="orange",
    label="Curated Ancestors- Curated Extants",
)
mutations = [101, 144]
for mutation in mutations:
    plt.axvline(x=mutation, color="r", linestyle="--", linewidth=1, alpha=1)

legend_elements = [
    Patch(facecolor="r", edgecolor="black", linestyle="--", label="mutagenesis sites"),
    Patch(
        facecolor="blue",
        edgecolor="black",
        linestyle="--",
        label="Ancestors to Extants ",
    ),
    Patch(
        facecolor="orange",
        edgecolor="black",
        linestyle="--",
        label="Curated ancestors to Curated Extants",
    ),
]

plt.legend(
    handles=legend_elements,
)
plt.ylabel("Wasserstein distance")
plt.xlabel("Sequence index")
plt.title("Wasserstein distance per column between Ancestors and Extants (GCN4)")
plt.show()


################### CURATED ANCESTORS & EXTANTS ##################
xticks = range(0, len(wd_curated_a_to_curated_e))

fig = plt.figure(figsize=(12, 8))
plt.plot(
    xticks[100:146],
    wd_a_to_e[100:146],
    alpha=1,
    color="blue",
    label="Ancestors-Extants",
)
plt.plot(
    xticks[100:146],
    wd_curated_a_to_curated_e[100:146],
    alpha=1,
    color="orange",
    label="Curated ancestors-Extants",
)


mutations = [101, 144]
for mutation in mutations:
    plt.axvline(x=mutation, color="r", linestyle="--", linewidth=1, alpha=1)

legend_elements = [
    Patch(facecolor="r", edgecolor="black", linestyle="--", label="mutagenesis sites"),
    Patch(
        facecolor="blue",
        edgecolor="black",
        linestyle="--",
        label="Ancestors to Extants ",
    ),
    Patch(
        facecolor="orange",
        edgecolor="black",
        linestyle="--",
        label="Curated ancestors to Curated Extants",
    ),
]

plt.legend(
    handles=legend_elements,
)
plt.ylabel("Wasserstein distance")
plt.xlabel("Sequence index")

plt.title("Wasserstein distance per column between Ancestors and Extants (GCN4)")
plt.show()
# -

# #### GCN4 - curated sequence weights distributions

# +

################ UNCURATED ##############

aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/gcn4/alns/"
gcn4_e = pd.read_pickle(aln_path + "gcn4_extants_no_dupes.pkl")
gcn4_a = pd.read_pickle(aln_path + "gcn4_ancestors_no_dupes.pkl")
gcn4_ae = pd.read_pickle(aln_path + "gcn4_ancestors_extants_no_dupes.pkl")

################### CURATED ANCESTORS & EXTANTS ##################
aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/gcn4/curated/alns/"
curated_gcn4_e = st.read_aln_file(aln_path + "gcn4_0.3_sim_extants_no_dupes.aln")
curated_gcn4_a = st.read_aln_file(aln_path + "gcn4_0.3_sim_ancestors_no_dupes.aln")
curated_gcn4_ae = st.read_aln_file(
    aln_path + "gcn4_0.3_sim_ancestors_0.3_sim_extants_no_dupes.aln"
)

datasets = [gcn4_e, gcn4_a, gcn4_ae, curated_gcn4_e, curated_gcn4_a, curated_gcn4_ae]

for data in datasets:
    numpy_aln, _, _ = st.convert_msa_numpy_array(data)
    data["seq_sim"] = st.reweight_by_seq_similarity(numpy_aln, 0.2)
    data["col_sim"] = st.position_based_seq_weighting(numpy_aln, 4)

# +

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.hist(np.log(gcn4_a["col_sim"]), color="red", alpha=1, bins=20, label="Ancestors")
ax1.hist(
    np.log(curated_gcn4_a["col_sim"]),
    color="blue",
    alpha=0.5,
    bins=20,
    label="Curated Ancestors",
)

ax1.set_ylabel("Count")
ax1.set_xlabel("ln(Sequence weight)")

ax1.legend()


ax2.hist(np.log(gcn4_e["col_sim"]), color="red", alpha=1, bins=20, label="Extants")
ax2.hist(
    np.log(curated_gcn4_e["col_sim"]),
    color="blue",
    alpha=0.5,
    bins=20,
    label="Curated extants",
)

ax2.set_ylabel("Count")
ax2.set_xlabel("ln(Sequence weight)")

ax2.set_title("GCN4 Sequence Weight Distributions")
ax2.legend()


ax3.hist(
    np.log(gcn4_ae["col_sim"]), color="red", alpha=1, bins=20, label="Ancestors/Extants"
)
ax3.hist(
    np.log(curated_gcn4_ae["col_sim"]),
    color="blue",
    alpha=0.5,
    bins=20,
    label="Curated Ancestors/Extants",
)
ax3.set_ylabel("Count")
ax3.set_xlabel("ln(Sequence weight)")

ax3.legend()

plt.show()
# -


# # MAFG

mafg_variants = pd.read_csv(
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/MAFG_MOUSE_Tsuboyama_2023_1K1V.csv"
)
mafg_variants.head()

# +
mafg_variants["DMS_score"].describe()

plt.hist(mafg_variants["DMS_score"], bins=40)
plt.ylabel("Number of variants")
plt.xlabel("Fitness")
plt.title("MAFG variants")
plt.show()

# +
aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/mafg_mouse/alns/"
gcn4_e = st.read_aln_file(aln_path + "mafg_extants_no_dupes.fasta")
gcn4_a = st.read_aln_file(aln_path + "mafg_ancestors_no_dupes.fasta")

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))

plot_entropy(
    ancestors=gcn4_a,
    extants=gcn4_e,
    anc_count=3682,
    ext_count=962,
    mutations=[0, 40],
    protein="MAFG",
    title="Column entropy ancestors vs extants",
    ax=ax1,
)


plt.show()
# -

# ### MAFG - Wasserstein Distance

# +
path = "/Users/sebs_mac/uni_OneDrive/honours/data/mafg_mouse/alns/"

wd_a_to_e, wd_a_to_ae, wd_e_to_ae, wd_a_to_v, wd_e_to_v, wd_ae_to_v = (
    calc_column_wasserstein_distances(
        ancestors=path + "mafg_ancestors_no_dupes.fasta",
        extants=path + "mafg_extants_no_dupes.fasta",
        ancestors_extants=path + "mafg_ancestors_extants_no_dupes.fasta",
        variants="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/MAFG_MOUSE_Tsuboyama_2023_1K1V.csv",
    )
)

print(f"Dist A-E: {np.mean(wd_a_to_e)}")
print(f"Dist A-AE: {np.mean(wd_a_to_ae)}")
print(f"Dist E-AE: {np.mean(wd_e_to_ae)}")

print(f"Dist A-V: {np.mean(wd_a_to_v)}")
print(f"Dist E-V: {np.mean(wd_e_to_v)}")
print(f"Dist AE-V: {np.mean(wd_ae_to_v)}")

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 6))

ax1.bar(
    np.arange(6),
    [
        np.mean(wd_a_to_e),
        np.mean(wd_a_to_ae),
        np.mean(wd_e_to_ae),
        np.mean(wd_a_to_v),
        np.mean(wd_e_to_v),
        np.mean(wd_ae_to_v),
    ],
)
ax1.set_title("MAFG: Uncurated Dataset comparisons")
ax1.set_ylabel("Wasserstein distance")
ax1.set_xticks(
    ticks=np.arange(6),
    labels=["Anc-Ex", "Anc-Anc/Ex", "E-Anc/Ex", "Anc-Var", "Ex-Var", "Anc/Ex-Var"],
    rotation=45,
)
ax1.set_ylim(0, 2)
ax1.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
plt.show()


# +
################ UNCURATED ##############
xticks = range(0, len(wd_a_to_e))

fig = plt.figure(figsize=(12, 8))
plt.plot(xticks, wd_a_to_e, alpha=1, color="blue", label="Ancestors-Extants")
mutations = [0, 40]
for mutation in mutations:
    plt.axvline(x=mutation, color="r", linestyle="--", linewidth=1, alpha=1)

legend_elements = [
    Patch(facecolor="r", edgecolor="black", linestyle="--", label="mutagenesis sites"),
    Patch(
        facecolor="blue",
        edgecolor="black",
        linestyle="--",
        label="Ancestors to Extants ",
    ),
]

plt.legend(
    handles=legend_elements,
)
plt.ylabel("Wasserstein distance")
plt.xlabel("Sequence index")
plt.title("Wasserstein distance per column between Ancestors and Extants (MAFG)")
plt.show()
# -

# ## Model extant recons compared to actual

import src.utils.seq_tools as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.utils.datasets import MSA_Dataset
from src.models.seqVAE import SeqVAE
import yaml
import torch
import src.utils.statistics as stats
import src.utils.visualisation as vs


# +
def perform_reconstruction(
    sequences: pd.DataFrame, model: SeqVAE, device
) -> pd.DataFrame:
    """
    Reconstructs sequences using a SeqVAE model.

    Args:
        sequences (pd.DataFrame): A DataFrame containing the sequences to be reconstructed.
        model (SeqVAE): The SeqVAE model used for reconstruction.
        device: The device (e.g., "cpu", "cuda") on which the model will be run.

    Returns:
        pd.DataFrame: A DataFrame containing the reconstructed sequences and their corresponding IDs.
    """
    encodings = sequences["sequence"].apply(st.seq_to_one_hot)
    dataset = MSA_Dataset(
        encodings, np.arange(sequences.shape[0]), extants["id"], device
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    recons = []
    ids = []
    for x, _, id in loader:
        x = x.to(device)
        indices = model.reconstruct(x)

        # how to have two loops in list comprehension
        sequences = [
            "".join([st.GAPPY_PROTEIN_ALPHABET[i] for i in seq]) for seq in indices
        ]
        recons.extend(sequences)
        ids.extend(id)

    df = pd.DataFrame({"id": ids, "sequence": recons})
    return df


def prepare_model(sequences: pd.DataFrame, state_dict: str, settings: dict) -> SeqVAE:

    seq_len = len(sequences["sequence"].values[0])
    input_dims = seq_len * st.GAPPY_ALPHABET_LEN

    device = torch.device("mps")
    model = SeqVAE(
        dim_latent_vars=settings["latent_dims"],
        dim_msa_vars=input_dims,
        num_hidden_units=settings["hidden_dims"],
        settings=settings,
        num_aa_type=21,
    )

    model.load_state_dict(torch.load(state_dict, map_location=torch.device("cpu")))
    model.to(device)
    model.eval()

    return model


# +

aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/all_alns/"
proteins = ["gb1", "gcn4", "gfp", "mafg", "a4"]

latent_dims = [8, 5, 3, 4, 7]

mutation_positions = [[264, 265, 266, 279], [101, 144], [3, 237], [0, 40], [671, 712]]

device = torch.device("mps")

for prot, mutations, dims in zip(proteins, mutation_positions, latent_dims):

    extants = st.read_aln_file(aln_path + f"{prot.upper()}_extants_no_dupes.aln")

    state_dict = f"/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/zero_shot_tasks/model_states/{prot}_ae_r1_model_state.pt"
    with open("../data/dummy_config.yaml", "r") as stream:
        settings = yaml.safe_load(stream)
    settings["latent_dims"] = dims

    model = prepare_model(extants, state_dict, settings)

    reconstructions = perform_reconstruction(extants, model, device)
    st.write_fasta_file(f"{prot}_ae_model_extant_reconstructions.aln", reconstructions)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    vs.plot_entropy(
        seqs_1=reconstructions,
        seqs_2=extants,
        mutations=mutations,
        protein=prot.upper(),
        title="Column entropy extants vs reconstructions",
        ax=ax1,
        label_1="Recons",
        label_2="Extants",
        max_entropy=2.5,
    )

    vs.plot_ppm_difference(extants, reconstructions, fig, ax2)
    ax2.set_title(f"{prot.upper()} Extant PPM - recons PPM ")
    ax2.set_xlabel("Sequence index")
    ax2.set_ylabel("Amino acid")

    plt.show()


# -

from ipydatagrid import DataGrid, TextRenderer, BarRenderer, Expr, ImageRenderer


# +
aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/all_alns/"
extants = st.read_aln_file(aln_path + "gfp_extants_no_dupes.aln")


state_dict = f"/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/zero_shot_tasks/model_states/gfp_ae_r1_model_state.pt"
with open("../data/dummy_config.yaml", "r") as stream:
    settings = yaml.safe_load(stream)
settings["latent_dims"] = 3

model = prepare_model(extants, state_dict, settings)

device = torch.device("mps")
encodings = extants["sequence"].apply(st.seq_to_one_hot)
dataset = MSA_Dataset(encodings, np.arange(extants.shape[0]), extants["id"], device)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


# +
import numpy as np
import pandas as pd
from ipywidgets import FloatSlider, Dropdown, ColorPicker, HBox, VBox
from ipydatagrid import DataGrid, TextRenderer, Expr
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import FloatSlider, Dropdown, ColorPicker, HBox, VBox, Output
from IPython.display import display


# Function to update the expression and DataGrid
def on_change(*args, **kwargs):

    conditional_expression.value = (
        "'{color}' if cell.value {operator} {highlight} else default_value".format(
            operator=operator_dropdown.value,
            highlight=highlight.value,
            color=output_colorpicker.value,
        )
    )


def update_probs(*args, **kwargs):

    log_p = model.latent_to_log_p(
        torch.Tensor(
            [
                reference_slider_1.value,
                reference_slider_2.value,
                reference_slider_3.value,
            ]
        ).to(device),
        238,
        21,
    )

    p = np.exp(log_p)
    huge_data, huge_df = update_datagrid(p)
    conditional_huge_datagrid.data = huge_df
    # update_3d_plot()


def update_datagrid(sequence_log_p: np.ndarray):

    to_plot = sequence_log_p[0, :, :]
    # transpose so rows are AA, cols are seq positions
    to_plot = to_plot.T

    data = {"data": [], "schema": {}}

    data["data"] = to_plot.tolist()

    data["schema"]["fields"] = [
        {"name": c, type: "number"} for c in st.GAPPY_PROTEIN_ALPHABET
    ]

    return data, pd.DataFrame(data["data"], index=st.GAPPY_PROTEIN_ALPHABET)


# Define the rendering function
def renderer_function(cell, default_value):
    return "#fc8403" if cell.value > 0.2 else default_value


# def update_3d_plot():
#     ax.clear()
#     ax.scatter(reference_slider_1.value, reference_slider_2.value, reference_slider_3.value, c='r', marker='o')
#     ax.set_xlabel('Reference 1')
#     ax.set_ylabel('Reference 2')
#     ax.set_zlabel('Reference 3')
#     fig.canvas.draw()


# Initial setup
x, _, _ = iter(loader).__next__()
log_p = model.get_log_p(x)
huge_data, huge_df = update_datagrid(np.exp(log_p))
conditional_expression = Expr(renderer_function)

default_renderer = TextRenderer(background_color=conditional_expression, format=".3f")
conditional_huge_datagrid = DataGrid(huge_df, default_renderer=default_renderer)

# Widgets
operator_dropdown = Dropdown(options=["<", ">"], value="<")
reference_slider_1 = FloatSlider(value=0.5, min=-10, max=10)
reference_slider_2 = FloatSlider(value=0.5, min=-10, max=10)
reference_slider_3 = FloatSlider(value=0.5, min=-10, max=10)
highlight = FloatSlider(value=0.5, min=0, max=1)
output_colorpicker = ColorPicker(value="#fc8403")


# Observe changes
operator_dropdown.observe(on_change, "value")
reference_slider_1.observe(update_probs, "value")
reference_slider_2.observe(update_probs, "value")
reference_slider_3.observe(update_probs, "value")
highlight.observe(on_change, "value")
output_colorpicker.observe(on_change, "value")

# 3D plot setup
output_3d = Output()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# Initial 3D plot
# update_3d_plot()

# Layout
hbox = HBox(
    (
        operator_dropdown,
        reference_slider_1,
        reference_slider_2,
        reference_slider_3,
        highlight,
        output_colorpicker,
    )
)
display(VBox([conditional_huge_datagrid, hbox]))

# -


log_p.shape
