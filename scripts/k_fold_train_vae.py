"""k_fold_train_vae.py"""

import torch.utils
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seq_trainer import seq_train
import pandas as pd
import evoVAE.utils.seq_tools as st
from datetime import datetime
import yaml, time, os, torch, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

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

def plot_losses(unique_id_path: str, fold: int):
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
        "-w",
        "--weight-decay",
        action="store",
        default=0.0,
        type=float,
        help="Weight decay. Defaults to zero",
    )

    parser.add_argument(
        "-l",
        "--latent-dims",
        action="store",
        default=3,
        type=int,
        help="Latent dims. Defaults to 3",
    )



    return parser.parse_args()


### MAIN PROGRAM ###
args = setup_parser()



# read in the config file
with open(args.config, "r") as stream:
    settings = yaml.safe_load(stream)

# If flag is included, zero shot prediction will occur.
# Must be specified in config.yaml
settings["weight_decay"] = args.weight_decay
settings["latent_dims"] = args.latent_dims
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
unique_id_path = f"{args.output}_r{args.replicate}"


logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(message)s",  # Format the log messages
    datefmt="%H:%M:%S",  # Date format
    filename=f"{unique_id_path}.log",  # Log file name
    filemode="w",  # Write mode (overwrites the log file each time the program runs)
)
logger = logging.getLogger(__name__)
logger.info(f"Run_id: {unique_id_path}")
logger.info(f"Start time: {datetime.now()}")

# save config for the run
logger.info("###CONFIG###")
yaml_str = yaml.dump(settings, default_flow_style=False)
logger.info(yaml_str)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# get the sequence length from first sequence
seq_len = len(aln["sequence"][0])
num_seq = aln.shape[0]
input_dims = seq_len * st.GAPPY_ALPHABET_LEN

logger.info(f"Seq length: {seq_len}")
logger.info(f"Original aln size : {num_seq}")

# work out each index for the k-folds
num_seq_subset = num_seq // args.folds + 1
idx_subset = []
np.random.seed(42)
random_idx = np.random.permutation(range(num_seq))
for i in range(args.folds):
    idx_subset.append(random_idx[i * num_seq_subset : (i + 1) * num_seq_subset])

logger.info(f"Number of folds : {args.folds}")
logger.info(f"Fold size: {len(idx_subset[0])}")

# hold ids and marginal probability of the validation set
fold_elbos = []
unique_name = []
for fold in range(args.folds):
    logger.info(f"Fold {fold + 1}")
    logger.info("-------")

    # save the final metrics to file.
    unique_name.append(f"{unique_id_path}_fold_{fold + 1}")

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
    logger.info("Training model")
    trained_model = seq_train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=settings,
        unique_id=f"{unique_id_path}_fold_{fold + 1}",
    )

    # plot the loss for visualtion of learning
    plot_losses(unique_id_path, fold)
    logger.info("Loss plotted")

    torch.save(
        trained_model.state_dict(),
        f"{unique_id_path}_fold_{fold + 1}_model_state.pt",
    )
    logger.info("Model saved")


    ### Estimate marginal probability assigned to validation sequences. ###
    # need to use batch size of one to allow for multiple samples of each data point
    recon_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    logger.info("Estimating marginal probability")
    trained_model.eval()
    with torch.no_grad():
        # can reuse the recon loader with a single batch size for multiple samples
        elbos = []
        for x, _, _ in recon_loader:
            log_elbo = trained_model.compute_elbo_with_multiple_samples(
                x, num_samples=5000
            )
            elbos.append(log_elbo.item())

        # get average log ELBO for the validation set
        mean_elbo = np.mean(elbos)
        fold_elbos.append(mean_elbo)

    logger.info(f"Elapsed time: {(time.time() - start) / 60} minutes")

# store our metrics
unique_name.extend(["mean", "std"]),
fold_elbos.extend([np.mean(fold_elbos), np.std(fold_elbos)])
all_metrics = pd.DataFrame(
    {
        "unique_id": unique_name,
        "marginal": fold_elbos,
    }
)

all_metrics.to_csv(
    f"{unique_id_path}_metrics.csv",
    index=False,
)

logger.info("###MODEL###")
logger.info(f"{str(model)}")
logger.info(f"Job length: {(time.time() - start) / 60} minutes")
