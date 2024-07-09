from evoVAE.utils.datasets import MSA_Dataset
import evoVAE.utils.seq_tools as st
from evoVAE.models.seqVAE import SeqVAE
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_mu(model, data_loader) -> pd.DataFrame:

    ids = []
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
    protein_name,
    tree_seq_path,
    ae_state_dict,
    a_state_dict,
    e_state_dict,
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

    tree_loader = torch.utils.data.DataLoader(tree_dataset, batch_size=1, shuffle=False)

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

    model.load_state_dict(torch.load(ae_state_dict, map_location=device))
    ae_latent = get_mu(model, tree_loader)
    ae_rgb = [
        rgb_to_hex_normalized(*((z - np.min(z)) / (np.max(z) - np.min(z))))
        for z in ae_latent["mu"]
    ]
    ae_latent["COLOR"] = ae_rgb
    ae_latent.drop(columns=["mu"], inplace=True)
    write_itol_dataset_symbol(f"{protein_name}_ae_model_itol.csv", ae_latent)

    model.load_state_dict(torch.load(a_state_dict, map_location=device))
    a_latent = get_mu(model, tree_loader)
    a_rgb = [
        rgb_to_hex_normalized(*((z - np.min(z)) / (np.max(z) - np.min(z))))
        for z in a_latent["mu"]
    ]
    a_latent["COLOR"] = a_rgb
    a_latent.drop(columns=["mu"], inplace=True)
    write_itol_dataset_symbol(f"{protein_name}_a_model_itol.csv", a_latent)

    model.load_state_dict(torch.load(e_state_dict, map_location=device))
    e_latent = get_mu(model, tree_loader)
    e_rgb = [
        rgb_to_hex_normalized(*((z - np.min(z)) / (np.max(z) - np.min(z))))
        for z in e_latent["z"]
    ]
    e_latent["COLOR"] = e_rgb
    e_latent.drop(columns=["mu"], inplace=True)
    write_itol_dataset_symbol(f"{protein_name}_e_model_itol.csv", e_latent)


def vis_tree(
    wt_id,
    tree_seq_path,
    state_dict,
    settings,
    title,
    n_samples=10,
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
        an_rgb = np.array(
            [
                rgb_to_hex_normalized(*((z - np.min(z)) / (np.max(z) - np.min(z))))
                for z in an_zs
            ]
        )
        ex_rgb = np.array(
            [
                rgb_to_hex_normalized(*((z - np.min(z)) / (np.max(z) - np.min(z))))
                for z in ex_zs
            ]
        )

    ax.scatter(
        an_zs[:, 0],
        an_zs[:, 1],
        an_zs[:, 2],
        c=an_rgb,
        label="Ancestors",
        s=50,
        marker=">",
        alpha=0.5,
    )
    ax.scatter(
        ex_zs[:, 0],
        ex_zs[:, 1],
        ex_zs[:, 2],
        c=ex_rgb,
        label="Extants",
        s=50,
        marker="o",
        alpha=0.5,
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
    if wt_id:
        wt_zs = np.array([z for z in wt["mu"]])

    if rgb:
        an_rgb = np.array(
            [
                rgb_to_hex_normalized(*((z - np.min(z)) / (np.max(z) - np.min(z))))
                for z in an_zs
            ]
        )
        ex_rgb = np.array(
            [
                rgb_to_hex_normalized(*((z - np.min(z)) / (np.max(z) - np.min(z))))
                for z in ex_zs
            ]
        )

    ax.scatter(
        np.vstack(ancestors["pca"].values)[:, 0],
        np.vstack(ancestors["pca"].values)[:, 1],
        c=an_rgb,
        label="Ancestors",
        s=50,
        marker=">",
        alpha=0.5,
    )
    ax.scatter(
        np.vstack(extants["pca"].values)[:, 0],
        np.vstack(extants["pca"].values)[:, 1],
        c=ex_rgb,
        label="Extants",
        s=50,
        marker="o",
        alpha=0.5,
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
