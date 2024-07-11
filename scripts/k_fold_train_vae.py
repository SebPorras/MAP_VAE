"""k_fold_train_vae.py"""

from random import shuffle
import torch.utils
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seq_trainer import seq_train, fitness_prediction
import pandas as pd
import evoVAE.utils.seq_tools as st
from evoVAE.trainer.seq_trainer import sample_latent_space
from datetime import datetime
import yaml, time, os, torch, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


CONFIG_FILE = 1
ARRAY_ID = 2
ALIGN_FILE = -2
MULTIPLE_REPS = 3
SEQ_LEN = 0
BATCH_ZERO = 0
SEQ_ZERO = 0
ALL_VARIANTS = 0

# errors
SUCCESS = 0
INVALID_FILE = 2


def prepare_dataset(
    original_aln: pd.DataFrame, subset_indices: np.array, device: torch.device
) -> MSA_Dataset:

    train_aln = original_aln.iloc[subset_indices].copy()
    # add weights to the sequences
    numpy_aln, _, _ = st.convert_msa_numpy_array(train_aln)
    weights = st.position_based_seq_weighting(
        numpy_aln, n_processes=int(os.getenv("SLURM_CPUS_PER_TASK"))
    )
    # weights = st.reweight_by_seq_similarity(numpy_aln, theta=0.2)
    train_aln["weights"] = weights

    # one-hot encode
    train_aln["encoding"] = train_aln["sequence"].apply(st.seq_to_one_hot)

    train_dataset = MSA_Dataset(
        train_aln["encoding"].to_numpy(),
        train_aln["weights"].to_numpy(),
        train_aln["id"],
        device,
    )

    return train_dataset


def validate_file(path):
    """Check that a valid file has been provided,
    otherwise exits with error code INVALID_FILE"""

    if (file := Path(path)).is_file():
        return file

    print(f"{path} is not a valid file. Aborting...")
    exit(INVALID_FILE)


def setup_parser() -> argparse.Namespace:
    """use argpase to sort CLI arguments and
    return the args."""

    parser = argparse.ArgumentParser(
        prog="Multiplxed Ancestral Phylogeny (MAP)",
        description="K-fold Train an instance of a VAE using ancestors",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=validate_file,
        required=True,
        action="store",
        metavar="config.yaml",
        help="A YAML file with required settings",
    )

    parser.add_argument(
        "-a",
        "--aln",
        action="store",
        metavar="example.aln",
        help="The alignment to train on in FASTA format",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="output",
        action="store",
        help="output directory. If not \
        specified, a directory called output will be created in the current working directory.",
    )

    parser.add_argument(
        "-r",
        "--replicate",
        default="1",
        action="store",
        help="specifies which replicate this run is. If a replicate indices file is provided \
        in the YAML config, this argument specifies which replicate column to use. Will default \
        to 1 if no argument is provided.",
    )

    parser.add_argument(
        "-f",
        "--folds",
        action="store",
        default=5,
        type=int,
        help="Number of k-folds. Defaults to 5 if not specified",
    )

    parser.add_argument(
        "-z",
        "--zero-shot",
        action="store_true",
        help="When specified, zero-shot prediction will be performed. This assumes you have \
            specified a DMS file and a corresponding metadata file",
    )

    return parser.parse_args()


### MAIN PROGRAM ###
args = setup_parser()

# read in the config file
with open(args.config, "r") as stream:
    settings = yaml.safe_load(stream)


# If flag is included, zero shot prediction will occur.
# Must be specified in config.yaml
settings["zero_shot"] = args.zero_shot
if args.zero_shot:
    dms_data = pd.read_csv(settings["dms_file"])
    one_hot = dms_data["mutated_sequence"].apply(st.seq_to_one_hot)
    dms_data["encoding"] = one_hot

    metadata = pd.read_csv(settings["dms_metadata"])
    metadata = metadata[metadata["DMS_id"] == settings["dms_id"]]

    # will hold our metrics
    unique_name = []
    spear = []
    k_recalls = []
    ndcgs = []
    roc = []


# overwrite the alignment in the config file
if args.aln is not None:
    settings["alignment"] = args.aln

# Read in the training dataset
if settings["alignment"].split(".")[-1] in ["fasta", "aln"]:
    aln = st.read_aln_file(settings["alignment"])
else:
    aln = pd.read_pickle(settings["alignment"])

start = time.time()

# unique identifier for this experiment
unique_id_path = Path(args.output + "_r" + args.replicate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# get the sequence length from first sequence
seq_len = len(aln["sequence"][0])
num_seq = aln.shape[0]
input_dims = seq_len * 21

log = ""
log += f"Seq length: {seq_len}\n"
log += f"Original aln size : {num_seq}\n"
print(log)

# work out each index for the k-folds
num_seq_subset = num_seq // args.folds + 1
idx_subset = []
random_idx = np.random.permutation(range(num_seq))
for i in range(args.folds):
    idx_subset.append(random_idx[i * num_seq_subset : (i + 1) * num_seq_subset])

log += f"Number of folds : {args.folds}\n"
log += f"Fold size: {len(idx_subset[0])}\n"
print(log)

for fold in range(args.folds):
    log += f"Fold {fold + 1}\n"
    log += "-------\n"

    val_idx = idx_subset[fold]
    train_idx = np.array(list(set(range(num_seq)) - set(val_idx)))

    # one-hot encodes and weights seqs before sending to device
    train_dataset = prepare_dataset(aln, train_idx, device)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
    )

    val_dataset = prepare_dataset(aln, val_idx, device)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=settings["batch_size"],
    )

    # instantiate the model
    model = SeqVAE(
        dim_latent_vars=settings["latent_dims"],
        dim_msa_vars=input_dims,
        num_hidden_units=settings["hidden_dims"],
        settings=settings,
        num_aa_type=settings["AA_count"],
    )
    model = model.to(device)

    # Training Loop
    trained_model = seq_train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=settings,
        unique_id=f"{unique_id_path}_fold_{fold + 1}",
    )

    # plot the loss for visualtion of learning
    losses = pd.read_csv(f"{unique_id_path}_fold_{fold + 1}_loss.csv")

    plt.figure(figsize=(12, 8))
    plt.plot(losses["epoch"], losses["elbo"], label="train", marker="o", color="b")
    plt.plot(
        losses["epoch"], losses["val_elbo"], label="validation", marker="x", color="r"
    )
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.legend()
    plt.title(f"{unique_id_path}_fold_{fold + 1}")
    plt.savefig(f"{unique_id_path}_fold_{fold + 1}_loss.png", dpi=300)

    torch.save(
        trained_model.state_dict(),
        f"{unique_id_path}_fold_{fold + 1}_model_state.pt",
    )

    if settings["zero_shot"]:

        # will also create a plot of actual vs predicted.
        # TODO: Add the code to scale up for different numbers of mutations
        spear_rho, k_recall, ndcg, roc_auc = fitness_prediction(
            trained_model,
            dms_data,
            metadata,
            f"{unique_id_path}_fold_{fold + 1}",
            device,
            mutation_count=ALL_VARIANTS,
            n_samples=5000,
        )

        # save the final metrics to file.
        unique_name.append(f"{unique_id_path}_fold_{fold + 1}")
        spear.append(spear_rho)
        k_recalls.append(k_recall)
        ndcgs.append(ndcg)
        roc.append(roc_auc)

    ### reconstruction of the validation set ###
    # need to use batch size of one to allow for multiple samples of each data point
    recon_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    ids, x_hats = sample_latent_space(
        model=trained_model, data_loader=recon_loader, num_samples=100
    )
    # save for later as we don't need the GPU
    recon = pd.DataFrame(
        {
            "id": ids,
            "sequence": aln[aln["id"].isin(ids)]["sequence"],
            "reconstruction": x_hats,
        }
    )
    recon.to_pickle(f"{unique_id_path}_fold_{fold + 1}_val_recons.pkl")

    ### Reconstruction of extant aln ###
    if settings["extant_aln"].split(".")[-1] in ["fasta", "aln"]:
        extant_aln = st.read_aln_file(settings["extant_aln"])
    else:
        extant_aln = pd.read_pickle(settings["extant_aln"])

    numpy_aln, _, _ = st.convert_msa_numpy_array(extant_aln)
    weights = st.position_based_seq_weighting(
        numpy_aln, n_processes=int(os.getenv("SLURM_CPUS_PER_TASK"))
    )
    # weights = st.reweight_by_seq_similarity(numpy_aln, theta=0.2)
    extant_aln["weights"] = weights
    extant_aln["encoding"] = extant_aln["sequence"].apply(st.seq_to_one_hot)

    extant_dataset = MSA_Dataset(
        extant_aln["encoding"].to_numpy(),
        extant_aln["weights"].to_numpy(),
        extant_aln["id"],
        device,
    )

    extant_loader = torch.utils.data.DataLoader(
        extant_dataset, batch_size=1, shuffle=False
    )
    ids, x_hats = sample_latent_space(
        model=trained_model, data_loader=extant_loader, num_samples=100
    )
    # save for later as we don't need the GPU
    recon = pd.DataFrame(
        {
            "id": ids,
            "sequence": extant_aln[extant_aln["id"].isin(ids)]["sequence"],
            "reconstruction": x_hats,
        }
    )
    recon.to_pickle(f"{unique_id_path}_fold_{fold + 1}_extant_recons.pkl")


print(len(unique_name), len(spear), len(k_recalls), len(ndcgs), len(roc))

# save config for the run
yaml_str = yaml.dump(settings, default_flow_style=False)
with open(f"{unique_id_path}_log.txt", "w") as file:
    file.write(f"Run_id: {unique_id_path}\n")
    file.write(f"Time: {datetime.now()}\n")
    file.write("###CONFIG###\n")
    file.write(f"{yaml_str}\n")
    file.write("###RUN INFO###\n")
    file.write(log)
    file.write("###MODEL###\n")
    file.write(f"{str(model)}\n")
    file.write("###TIME###\n")
    file.write(f"{(time.time() - start) / 60} minutes\n")

# store our metrics
all_metrics = pd.DataFrame(
    {
        "unique_id": unique_name,
        "spearman_rho": spear,
        "top_k_recall": k_recalls,
        "ndcg": ndcgs,
        "roc_auc": roc,
    }
)

all_metrics.to_csv(
    f"{unique_id_path}_metrics.csv",
    index=False,
)
