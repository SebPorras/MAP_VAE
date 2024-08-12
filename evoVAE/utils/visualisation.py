from evoVAE.utils.datasets import MSA_Dataset
import evoVAE.utils.seq_tools as st
from evoVAE.models.seqVAE import SeqVAE
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import evoVAE.utils.statistics as stats
from matplotlib.patches import Patch
from typing import List, Optional
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns


@torch.no_grad()
def get_mu(model: SeqVAE, data_loader) -> pd.DataFrame:

    ids = []
    model.eval()
    for x, _, name in data_loader:
        x = torch.flatten(x, start_dim=1)
        mu, sigma = model.encoder(x)

    ids.extend(name)
    mu = mu.cpu().data.numpy().tolist()

    id_to_mu = pd.DataFrame({"id": ids, "mu": mu})

    return id_to_mu


def rgb_to_hex_normalized(r, g, b):
    # Scale the values from 0-1 to 0-255
    r_scaled = int(r * 255)
    g_scaled = int(g * 255)
    b_scaled = int(b * 255)

    # Convert to hexadecimal and combine
    return "#{0:02X}{1:02X}{2:02X}".format(r_scaled, g_scaled, b_scaled)


def write_itol_dataset_symbol(
    filename: str,
    data: pd.DataFrame,
    symbol: int = 2,
    size: int = 2,
    fill: float = 1.0,
    position: float = 1.0,
    label: float = 0.5,
):

    ids = [x.split("_")[0] if "tree" in x else x for x in data["id"]]
    data["NODE_ID"] = ids

    nodes = [symbol] * len(data)
    data["SYMBOL"] = nodes

    nodes = [size] * len(data)
    data["SIZE"] = nodes

    nodes = [fill] * len(data)
    data["FILL"] = nodes

    nodes = [position] * len(data)
    data["POSITION"] = nodes

    with open(filename, "w") as file:

        print("DATASET_SYMBOL", file=file)
        print("SEPARATOR COMMA", file=file)
        print("DATASET_LABEL,example symbols", file=file)
        print("COLOR,#ffff00", file=file)
        print("DATA", file=file)
        print("#NODE_ID, SYMBOL, SIZE, COLOR, FILL, POSITION", file=file)

        for _, row in data.iterrows():
            file.write(
                f"{row['NODE_ID']},{row['SYMBOL']},{row['SIZE']},{row['COLOR']},{row['FILL']},{row['POSITION']}\n"
            )


def get_model_embeddings(
    path,
    ae_file_name,
    variant_path,
    ae_state_dict,
    a_state_dict,
    e_state_dict,
    settings,
    n_samples=10,
):

    if ae_file_name.split(".")[-1] in ["fasta", "aln"]:
        ae = st.read_aln_file(path + ae_file_name)
    else:
        ae = pd.read_pickle(path + ae_file_name)

    variants = pd.read_csv(variant_path)
    variants.drop(columns=["DMS_score", "DMS_score_bin"], inplace=True)
    variants.rename(
        columns={"mutant": "id", "mutated_sequence": "sequence"}, inplace=True
    )

    ae_vars = pd.concat([ae, variants], ignore_index=True)

    one_hot = ae_vars["sequence"].apply(st.seq_to_one_hot)
    ae_vars["encoding"] = one_hot

    device = torch.device("mps")
    ae_var_dataset = MSA_Dataset(
        ae_vars["encoding"],
        pd.Series(np.arange(len(one_hot))),
        ae_vars["id"],
        device=device,
    )

    num_seqs = len(ae_vars["sequence"])
    ae_var_loader = torch.utils.data.DataLoader(
        ae_var_dataset, batch_size=num_seqs, shuffle=False
    )

    seq_len = ae_var_dataset[0][0].shape[0]
    input_dims = seq_len * settings["AA_count"]

    model = SeqVAE(
        dim_latent_vars=settings["latent_dims"],
        dim_msa_vars=input_dims,
        num_hidden_units=settings["hidden_dims"],
        settings=settings,
        num_aa_type=settings["AA_count"],
    )

    model = model.to(device)

    model.load_state_dict(torch.load(ae_state_dict, map_location=device))
    # ae_latent = get_mean_z(model, ae_var_loader, device, n_samples)
    ae_latent = get_mu(model, ae_var_loader)

    model.load_state_dict(torch.load(a_state_dict, map_location=device))
    # a_latent = get_mean_z(model, ae_var_loader, device, n_samples)
    a_latent = get_mu(model, ae_var_loader)

    model.load_state_dict(torch.load(e_state_dict, map_location=device))
    # e_latent = get_mean_z(model, ae_var_loader, device, n_samples)
    e_latent = get_mu(model, ae_var_loader)

    return ae_latent, a_latent, e_latent


def latent_tree_to_itol(
    filename,
    tree_seq_path,
    state_dict,
    settings,
) -> None:

    device = torch.device("mps")
    tree_seqs = st.read_aln_file(tree_seq_path)

    one_hot = tree_seqs["sequence"].apply(st.seq_to_one_hot)
    tree_seqs["encoding"] = one_hot

    tree_dataset = MSA_Dataset(
        tree_seqs["encoding"],
        pd.Series(np.arange(len(one_hot))),
        tree_seqs["id"],
        device=device,
    )

    tree_loader = torch.utils.data.DataLoader(
        tree_dataset, batch_size=len(tree_dataset), shuffle=False
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

    model = model.to(device)

    model.load_state_dict(torch.load(state_dict, map_location=device))
    latent = get_mu(model, tree_loader)
    ae_rgb = [
        rgb_to_hex_normalized(*((z - np.min(z)) / (np.max(z) - np.min(z))))
        for z in latent["mu"]
    ]
    latent["COLOR"] = ae_rgb
    latent.drop(columns=["mu"], inplace=True)
    write_itol_dataset_symbol(f"{filename}_itol.csv", latent)


def vis_tree(
    wt_id,
    tree_seq_path,
    state_dict,
    settings,
    title,
    rgb=True,
    lower_2d=False,
) -> None:

    tree_seqs = st.read_aln_file(tree_seq_path)

    one_hot = tree_seqs["sequence"].apply(st.seq_to_one_hot)
    tree_seqs["encoding"] = one_hot
    num_seqs = len(tree_seqs["sequence"])
    device = torch.device("mps")

    tree_dataset = MSA_Dataset(
        tree_seqs["encoding"],
        np.arange(len(one_hot)),
        tree_seqs["id"],
        device=device,
    )

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

    model = model.to(device)

    model.load_state_dict(torch.load(state_dict, map_location=device))

    # latent = get_mean_z(model, tree_loader, device, n_samples)
    latent = get_mu(model, tree_loader)

    if lower_2d:
        fig, (ax) = plt.subplots(1, 1, figsize=(12, 8))
        tree_vis_2d(latent, wt_id, rgb=rgb, ax=ax)
    else:
        fig, (ax) = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={"projection": "3d"})
        tree_vis_3d(latent, wt_id, rgb=rgb, ax=ax)

    # ax.view_init(elev=30, azim=40)  # Change these values to get the desired orientation
    ax.set_title(f"{title}")
    ax.legend()
    plt.show()


def tree_vis_3d(data, wt_id, rgb, ax):

    ancestors = data[data["id"].str.contains("tree")]
    extants = data[~data["id"].str.contains("tree")]
    if wt_id:
        wt = data[data["id"] == wt_id]

    ax.set_xlabel("Z1")
    ax.set_ylabel("Z2")
    ax.set_zlabel("Z3")

    an_zs = np.array([z for z in ancestors["mu"]])
    ex_zs = np.array([z for z in extants["mu"]])
    if wt_id:
        wt_zs = np.array([z for z in wt["mu"]])

    # # Plot ancestors
    an_rgb = "red"
    ex_rgb = "blue"
    if rgb:

        min_r = np.min(np.stack(data["mu"])[:, 0])
        max_r = np.max(np.stack(data["mu"])[:, 0])
        min_g = np.min(np.stack(data["mu"])[:, 1])
        max_g = np.max(np.stack(data["mu"])[:, 1])
        min_b = np.min(np.stack(data["mu"])[:, 2])
        max_b = np.max(np.stack(data["mu"])[:, 2])

        an_rgb = [
            (
                (r - min_r) / (max_r - min_r),
                (g - min_g) / (max_g - min_g),
                (b - min_b) / (max_b - min_b),
            )
            for r, g, b in zip(
                np.stack(an_zs)[:, 0],
                np.stack(an_zs)[:, 1],
                np.stack(an_zs)[:, 2],
            )
        ]

        # an_rgb = np.array([rgb_to_hex_normalized(*x) for x in scaled_an_rgb])
        ex_rgb = np.array(
            [
                (
                    (r - min_r) / (max_r - min_r),
                    (g - min_g) / (max_g - min_g),
                    (b - min_b) / (max_b - min_b),
                )
                for r, g, b in zip(
                    np.stack(ex_zs)[:, 0],
                    np.stack(ex_zs)[:, 1],
                    np.stack(ex_zs)[:, 2],
                )
            ]
        )

        # ex_rgb = np.array([rgb_to_hex_normalized(*x) for x in scaled_ex_rgb])

    ax.scatter(
        an_zs[:, 0],
        an_zs[:, 1],
        an_zs[:, 2],
        c=an_rgb,
        label="Ancestors",
        s=50,
        marker=">",
        alpha=1,
    )
    ax.scatter(
        ex_zs[:, 0],
        ex_zs[:, 1],
        ex_zs[:, 2],
        c=ex_rgb,
        label="Extants",
        s=50,
        marker="o",
        alpha=1,
    )
    if wt_id:
        ax.scatter(
            wt_zs[:, 0],
            wt_zs[:, 1],
            wt_zs[:, 2],
            c="black",
            label="WT",
            s=500,
            marker="*",
            alpha=1,
        )


def tree_vis_2d(data, wt_id, rgb, ax):

    pca = PCA(n_components=2)

    zs_2d = pca.fit_transform(np.vstack(data["mu"].values))

    data["pca"] = list(zs_2d)

    # Filter ancestors and extants
    ancestors = data[data["id"].str.contains("tree")]
    extants = data[~data["id"].str.contains("tree")]

    if wt_id:
        wt = data[data["id"] == wt_id]

    an_rgb = "red"
    ex_rgb = "blue"

    an_zs = np.array([z for z in ancestors["mu"]])
    ex_zs = np.array([z for z in extants["mu"]])

    if rgb:

        min_r = np.min(np.stack(data["mu"])[:, 0])
        max_r = np.max(np.stack(data["mu"])[:, 0])
        min_g = np.min(np.stack(data["mu"])[:, 1])
        max_g = np.max(np.stack(data["mu"])[:, 1])
        min_b = np.min(np.stack(data["mu"])[:, 2])
        max_b = np.max(np.stack(data["mu"])[:, 2])

        an_rgb = [
            (
                (r - min_r) / (max_r - min_r),
                (g - min_g) / (max_g - min_g),
                (b - min_b) / (max_b - min_b),
            )
            for r, g, b in zip(
                np.stack(an_zs)[:, 0],
                np.stack(an_zs)[:, 1],
                np.stack(an_zs)[:, 2],
            )
        ]
        # an_rgb = np.array([rgb_to_hex_normalized(*x) for x in scaled_an_rgb])

        ex_rgb = np.array(
            [
                (
                    (r - min_r) / (max_r - min_r),
                    (g - min_g) / (max_g - min_g),
                    (b - min_b) / (max_b - min_b),
                )
                for r, g, b in zip(
                    np.stack(ex_zs)[:, 0],
                    np.stack(ex_zs)[:, 1],
                    np.stack(ex_zs)[:, 2],
                )
            ]
        )

        # ex_rgb = np.array([rgb_to_hex_normalized(*x) for x in scaled_ex_rgb])

    ax.scatter(
        np.vstack(ancestors["pca"].values)[:, 0],
        np.vstack(ancestors["pca"].values)[:, 1],
        c=an_rgb,
        label="Ancestors",
        s=50,
        marker=">",
        alpha=1,
    )
    ax.scatter(
        np.vstack(extants["pca"].values)[:, 0],
        np.vstack(extants["pca"].values)[:, 1],
        c=ex_rgb,
        label="Extants",
        s=50,
        marker="o",
        alpha=1,
    )
    if wt_id:
        ax.scatter(
            np.vstack(wt["pca"].values)[:, 0],
            np.vstack(wt["pca"].values)[:, 1],
            c="black",
            label="WT",
            s=200,
            marker="*",
            alpha=1,
        )

    ax.set_xlabel(f"PC1: {np.round(pca.explained_variance_ratio_[0] * 100, 2)}%")
    ax.set_ylabel(f"PC2 {np.round(pca.explained_variance_ratio_[1] * 100, 2)}%")


def visualise_variants(
    settings,
    variant_data,
    state_dict,
    title,
    seqs=None,
    wt_id=None,
    vis_2D=False,
    frac=1,
    aln=False,
):

    if aln:
        variants = st.read_aln_file(variant_data)
        variants.rename(
            columns={"sequence": "mutated_sequence", "id": "mutant"}, inplace=True
        )
    else:
        variants = pd.read_csv(variant_data)

    variants = variants.sample(frac=frac, random_state=42)

    variants["encoding"] = variants["mutated_sequence"].apply(st.seq_to_one_hot)

    if wt_id is not None:
        tree_seqs = st.read_aln_file(seqs)
        tree_seqs["encoding"] = tree_seqs["sequence"].apply(st.seq_to_one_hot)
        tree_seqs.rename(
            columns={"sequence": "mutated_sequence", "id": "mutant"}, inplace=True
        )
        tree_seqs = tree_seqs[tree_seqs["mutant"] == wt_id]

        variants = pd.concat([variants, tree_seqs])

    num_seqs = len(variants["mutated_sequence"])
    device = torch.device("mps")

    var_dataset = MSA_Dataset(
        variants["encoding"],
        np.arange(len(variants["encoding"])),
        variants["mutant"],
        device=device,
    )

    loader = torch.utils.data.DataLoader(
        var_dataset, batch_size=num_seqs, shuffle=False
    )
    seq_len = var_dataset[0][0].shape[0]

    input_dims = seq_len * settings["AA_count"]

    model = SeqVAE(
        dim_latent_vars=settings["latent_dims"],
        dim_msa_vars=input_dims,
        num_hidden_units=settings["hidden_dims"],
        settings=settings,
        num_aa_type=settings["AA_count"],
    )

    model.load_state_dict(torch.load(state_dict, map_location=device))
    model = model.to(device)

    if vis_2D:
        vis_variants_2d(model, title, variants, loader, wt_id)
    else:
        vis_variants_3d(model, title, variants, loader, wt_id)


def vis_variants_3d(model, title, variant_data, variant_loader, wt_id):

    id_to_mu = get_mu(model, variant_loader)
    id_to_mu.rename(columns={"id": "mutant"}, inplace=True)
    merged = variant_data.merge(id_to_mu, on="mutant")

    vars = merged[~merged["mutant"].str.contains(wt_id)]
    wt = merged[merged["mutant"].str.contains(wt_id)]

    variant_mus = np.stack(vars["mu"])
    wt_mus = np.stack(wt["mu"])

    fig, (ax) = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={"projection": "3d"})
    scatter = ax.scatter(
        variant_mus[:, 0],
        variant_mus[:, 1],
        variant_mus[:, 2],
        c=vars["DMS_score"],
        cmap="inferno",
        alpha=0.8,
    )

    wt_scatter = ax.scatter(
        wt_mus[:, 0],
        wt_mus[:, 1],
        wt_mus[:, 2],
        c="black",
        label="WT",
        s=200,
        marker="*",
        alpha=1,
    )

    ax.set_title(title)
    ax.set_xlabel("Z1")
    ax.set_ylabel("Z2")
    ax.set_zlabel("Z3")
    # Add color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Fitness Score")
    ax.legend()

    plt.show()


def vis_variants_2d(model, title, variant_data, variant_loader, wt_id):

    id_to_mu = get_mu(model, variant_loader)
    id_to_mu.rename(columns={"id": "mutant"}, inplace=True)
    merged = variant_data.merge(id_to_mu, on="mutant")

    pca = PCA(n_components=2)

    zs_2d = pca.fit_transform(np.vstack(merged["mu"].values))

    vars = merged[~merged["mutant"].str.contains(wt_id)]
    wt = merged[merged["mutant"].str.contains(wt_id)]
    merged["pca"] = list(zs_2d)

    # scaler = MinMaxScaler()
    # dms_values = scaler.fit_transform(merged["DMS_score"].values.reshape(-1, 1))

    fig, (ax) = plt.subplots(1, 1, figsize=(12, 8))
    scatter = ax.scatter(
        np.vstack(merged["pca"].values)[:, 0],
        np.vstack(merged["pca"].values)[:, 1],
        c=merged["DMS_score"],
        cmap="inferno",
    )

    ax.set_title(title)
    ax.set_xlabel(f"PCA1 ({round(pca.explained_variance_[0] * 100, 2)}%)")
    ax.set_ylabel(f"PCA2 ({round(pca.explained_variance_[1] * 100, 2)}%)")

    # Add color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Fitness Score")

    plt.show()


import matplotlib.pyplot as plt


def plot_entropy_ancestors(
    ancestors: List[str],
    extants: List[str],
    anc_count: int,
    ext_count: int,
    mutations: List[int],
    protein: str,
    title: str,
    ax: plt.Axes,
    start: int = 0,
    end: Optional[int] = None,
    max_entropy: float = 3,
) -> None:
    """
    Plots the entropy of ancestors and extants over a sequence.

    Args:
        ancestors (list): List of ancestor sequences.
        extants (list): List of extant sequences.
        anc_count (int): Number of ancestor sequences.
        ext_count (int): Number of extant sequences.
        mutations (list): List of mutation positions.
        protein (str): Name of the protein.
        title (str): Title of the plot.
        ax (matplotlib.axes.Axes): The axes on which to plot.
        start (int, optional): Starting position of the sequence. Defaults to 0.
        end (int, optional): Ending position of the sequence. Defaults to None.
        max_entropy (float, optional): Maximum entropy value for the y-axis. Defaults to 3.

    Returns:
        None
    """
    e_col_entropy = stats.calc_shannon_entropy(extants)
    a_col_entropy = stats.calc_shannon_entropy(ancestors)

    if end is not None:
        # across entire sequence
        xticks = range(start, end)
        e_col_entropy = e_col_entropy[start:end]
        a_col_entropy = a_col_entropy[start:end]
    else:
        xticks = range(start, len(a_col_entropy))

    fig = plt.figure(figsize=(12, 8))
    ax.plot(xticks, a_col_entropy, alpha=0.5, color="orange")
    ax.plot(xticks, e_col_entropy, alpha=0.5, color="blue")
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

    for mutation in mutations:
        ax.axvline(x=mutation, color="r", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_title(f"{protein}: {title}")


def plot_entropy(
    seqs_1: List[str],
    seqs_2: List[str],
    mutations: List[int],
    protein: str,
    title: str,
    ax: plt.Axes,
    start: int = 0,
    end: Optional[int] = None,
    max_entropy: float = 3,
    label_1: str = "1",
    label_2: str = "2",
) -> None:
    """
    Plots the entropy of ancestors and extants over a sequence.

    Args:
        ancestors (list): List of ancestor sequences.
        extants (list): List of extant sequences.
        anc_count (int): Number of ancestor sequences.
        ext_count (int): Number of extant sequences.
        mutations (list): List of mutation positions.
        protein (str): Name of the protein.
        title (str): Title of the plot.
        ax (matplotlib.axes.Axes): The axes on which to plot.
        start (int, optional): Starting position of the sequence. Defaults to 0.
        end (int, optional): Ending position of the sequence. Defaults to None.
        max_entropy (float, optional): Maximum entropy value for the y-axis. Defaults to 3.

    Returns:
        None
    """
    seqs_1_entropy = stats.calc_shannon_entropy(seqs_1)
    seqs_2_entropy = stats.calc_shannon_entropy(seqs_2)

    if end is not None:
        # across entire sequence
        xticks = range(start, end)
        seqs_1_entropy = seqs_1_entropy[start:end]
        seqs_2_entropy = seqs_2_entropy[start:end]
    else:
        xticks = range(start, len(seqs_2_entropy))

    fig = plt.figure(figsize=(12, 8))
    ax.plot(xticks, seqs_1_entropy, alpha=0.5, color="blue")
    ax.plot(xticks, seqs_2_entropy, alpha=0.5, color="orange")
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
            label=f"{protein} {label_2}",
        ),
        Patch(
            facecolor="blue",
            edgecolor="black",
            label=f"{protein} {label_1}",
        ),
    ]

    ax.legend(
        handles=legend_elements,
    )

    for mutation in mutations:
        ax.axvline(x=mutation, color="r", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_title(f"{protein}: {title}")


def plot_ppm_difference(
    extants, ancestors, fig, ax, xmax=None, xmin=0, ymin=0, ymax=21, pseudo=1
):
    """
    Plot the difference between two position probability matrices (PPMs).

    Parameters:
    - extants (list): List of extant sequences.
    - ancestors (list): List of ancestor sequences.
    - xmax (int, optional): Maximum x-axis value. Defaults to None.
    - xmin (int, optional): Minimum x-axis value. Defaults to 0.
    - ymin (int, optional): Minimum y-axis value. Defaults to 0.
    - ymax (int, optional): Maximum y-axis value. Defaults to 21.

    Returns:
    None
    """

    extant_ppm = np.log2(stats.calc_position_prob_matrix(extants, pseudo))
    ancestor_ppm = np.log2(stats.calc_position_prob_matrix(ancestors, pseudo))
    out = extant_ppm - ancestor_ppm

    cmap = sns.diverging_palette(240, 20, as_cmap=True)
    sns.heatmap(
        out,
        cmap=cmap,
        center=0,
        ax=ax,
        cbar=True,
        yticklabels=st.GAPPY_PROTEIN_ALPHABET,
    )

    # weird fix to set roation to 360 to get the y-axis labels to be horizontal
    ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
    if xmax is None:
        xmax = extant_ppm.shape[1]
    ax.set_xlim(xmin, xmax)
    xticks = np.arange(xmin, xmax, step=max(1, (xmax - xmin) // 10))
    ax.set_xticks(xticks)

    ax.set_xticklabels(xticks, rotation=45)

    # ax.set_ylim(ymin, ymax)
    # ax.set_yticks(range(21), st.GAPPY_PROTEIN_ALPHABET)
    # ax.set_yticklabels(st.GAPPY_PROTEIN_ALPHABET, rotation=360)
