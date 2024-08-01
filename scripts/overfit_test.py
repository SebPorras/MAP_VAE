"""train_vae.py"""

from random import shuffle
from sklearn.model_selection import train_test_split
import torch.utils
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seq_trainer import (
    train_loop,
    validation_loop,
)
import pandas as pd
import evoVAE.utils.seq_tools as st
from datetime import datetime
import yaml, time, os, torch, argparse
from pathlib import Path
import numpy as np
from evoVAE.loss.standard_loss import frange_cycle_linear
import matplotlib.pyplot as plt
import logging

FOLDS = 5
CONFIG_FILE = 1
ARRAY_ID = 2
ALIGN_FILE = -2
MULTIPLE_REPS = 3
SEQ_LEN = 0
BATCH_ZERO = 0
SEQ_ZERO = 0

# errors
SUCCESS = 0
INVALID_FILE = 2
ALL_VARIANTS = 0


def prepare_indexed_dataset(
    original_aln: pd.DataFrame, subset_indices: np.array, device: torch.device
) -> MSA_Dataset:

    train_aln = original_aln.iloc[subset_indices].copy()
    # add weights to the sequences
    numpy_aln, _, _ = st.convert_msa_numpy_array(train_aln)
    weights = st.position_based_seq_weighting(
        numpy_aln, n_processes=2
    )  # int(os.getenv("SLURM_CPUS_PER_TASK"))

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


def prepare_dataset(original_aln: pd.DataFrame, device: torch.device) -> MSA_Dataset:

    # add weights to the sequences
    numpy_aln, _, _ = st.convert_msa_numpy_array(original_aln)
    weights = st.position_based_seq_weighting(
        numpy_aln, n_processes=2
    )  # int(os.getenv("SLURM_CPUS_PER_TASK"))
    # )
    # weights = st.reweight_by_seq_similarity(numpy_aln, theta=0.2)
    original_aln["weights"] = weights

    # one-hot encode
    original_aln["encoding"] = original_aln["sequence"].apply(st.seq_to_one_hot)

    train_dataset = MSA_Dataset(
        original_aln["encoding"].to_numpy(),
        original_aln["weights"].to_numpy(),
        original_aln["id"],
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


def estimate_marginal(
    model: SeqVAE, loader: torch.utils.data.DataLoader, num_samples: int = 5000
) -> float:
    """
    Estimates the marginal likelihood of a given model using the evidence lower bound (ELBO) method.

    Args:
        model (SeqVAE): The trained sequence variational autoencoder model.
        loader (torch.utils.data.DataLoader): The data loader for the dataset.
        num_samples (int, optional): The number of samples to use for estimating the marginal likelihood.
            Defaults to 5000.

    Returns:
        float: The estimated marginal likelihood.

    """
    model.eval()
    elbos = []
    with torch.no_grad():
        for x, _, _ in loader:
            log_elbo = model.compute_elbo_with_multiple_samples(x, num_samples)
            elbos.append(log_elbo.item())

    # get average log ELBO for the validation set
    return np.mean(elbos)


# work out each index for the k-folds
def get_k_fold_indices(num_seqs, folds):

    num_seq_subset = num_seqs // folds + 1
    idx_subset = []
    np.random.seed(42)
    random_idx = np.random.permutation(range(num_seqs))

    for i in range(folds):
        idx_subset.append(random_idx[i * num_seq_subset : (i + 1) * num_seq_subset])

    return idx_subset


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
        help="Number of latent dimensions. Defaults to 3",
    )

    return parser.parse_args()


if __name__ == "__main__":

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the sequence length from first sequence
    seq_len = len(aln["sequence"].values[0])
    num_seq = aln.shape[0]
    input_dims = seq_len * st.GAPPY_ALPHABET_LEN

    logger.info(f"Alignment: {args.aln}")
    logger.info(f"Seq length: {seq_len}")
    logger.info(f"Original aln size : {num_seq}")
    logger.info(f"Using device: {device}")

    # # one-hot encodes and weights seqs before sending to device
    ancestors = aln[aln["id"].str.contains("tree")]
    ancestors = ancestors.reset_index(drop=True)

    extants = aln[~aln["id"].str.contains("tree")]
    extants = extants.reset_index(drop=True)

    anc_idxs = get_k_fold_indices(ancestors.shape[0], FOLDS)
    ext_idxs = get_k_fold_indices(extants.shape[0], FOLDS)
    unique_names = []
    train_margs = []
    anc_margs = []
    ext_margs = []
    for fold in range(FOLDS):
        logger.info(f"Fold {fold + 1}")
        logger.info("-------")

        unique_names.append(f"{unique_id_path}_fold_{fold + 1}")

        # create the fold indices
        anc_test_idx = anc_idxs[fold]
        logger.info(f"Ancestor test size {len(anc_test_idx)}")
        anc_train_idx = np.array(
            list(set(range(ancestors.shape[0])) - set(anc_test_idx))
        )

        ext_test_idx = ext_idxs[fold]
        logger.info(f"Extant test size {len(ext_test_idx)}")
        ext_train_idx = np.array(list(set(range(extants.shape[0])) - set(ext_test_idx)))

        # validation folds
        ext_test_dataset = prepare_indexed_dataset(extants, ext_test_idx, device)
        anc_test_dataset = prepare_indexed_dataset(ancestors, anc_test_idx, device)

        # training data
        training = pd.concat(
            [ancestors.iloc[anc_train_idx], extants.iloc[ext_train_idx]]
        )
        logger.info(f"Training size {training.shape[0]}")
        train_dataset = prepare_dataset(training, device)

        # make the loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=settings["batch_size"],
        )

        ext_test_loader = torch.utils.data.DataLoader(
            ext_test_dataset,
            batch_size=settings["batch_size"],
        )

        anc_test_loader = torch.utils.data.DataLoader(
            anc_test_dataset,
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

        optimiser = model.configure_optimiser(
            learning_rate=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=settings["epochs"]
        )
        anneal_schedule = frange_cycle_linear(settings["epochs"])
        with open(unique_id_path + f"_fold_{fold + 1}_loss.csv", "w") as file:
            file.write("epoch,train_elbo,anc_elbo,ext_elbo\n")
            file.flush()

            for epoch in range(settings["epochs"]):

                elbo, _, _ = train_loop(
                    model, train_loader, optimiser, epoch, anneal_schedule, scheduler
                )

                _, anc_elbo, _, _ = validation_loop(
                    model, anc_test_loader, epoch, anneal_schedule, None
                )

                _, ext_elbo, _, _ = validation_loop(
                    model, ext_test_loader, epoch, anneal_schedule, None
                )

                file.write(f"{epoch},{elbo},{anc_elbo},{ext_elbo}\n")
                file.flush()

        # plot the loss for visualtion of learning
        losses = pd.read_csv(f"{unique_id_path}_fold_{fold + 1}_loss.csv")
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
        plt.figure(figsize=(12, 8))
        plt.plot(
            losses["epoch"],
            losses["train_elbo"],
            label="Anc/Ext train",
            marker="o",
            color="b",
        )
        plt.plot(
            losses["epoch"],
            losses["anc_elbo"],
            label="Ancestor test",
            marker="x",
            color="r",
        )

        plt.plot(
            losses["epoch"],
            losses["ext_elbo"],
            label="Extant test",
            marker=">",
            color="g",
        )

        plt.xlabel("Epoch")
        plt.ylabel("ELBO")
        plt.legend()
        plt.title(f"{unique_id_path}_fold_{fold + 1}")
        plt.savefig(f"{unique_id_path}_fold_{fold + 1}_loss.png", dpi=300)

        # Estimate marginal probability assigned to validation sequences.
        logger.debug("Estimating marginal probability\n")

        train_marg_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False
        )
        train_marg = estimate_marginal(model, train_marg_loader, num_samples=5000)

        anc_test_marg_loader = torch.utils.data.DataLoader(
            anc_test_dataset, batch_size=1, shuffle=False
        )
        anc_marg = estimate_marginal(model, anc_test_marg_loader, num_samples=5000)

        ext_test_marg_loader = torch.utils.data.DataLoader(
            ext_test_dataset, batch_size=1, shuffle=False
        )
        ext_marg = estimate_marginal(model, ext_test_marg_loader, num_samples=5000)

        train_margs.append(train_marg)
        anc_margs.append(anc_marg)
        ext_margs.append(ext_marg)

        # calculations end here
        logger.debug(f"Elapsed time: {(time.time() - start) / 60} minutes\n")

    # # store our metrics
    all_metrics = pd.DataFrame(
        {
            "unique_id": unique_names,
            "train_marginal": train_margs,
            "anc_marginal": anc_margs,
            "ext_marginal": ext_margs,
        }
    )

    all_metrics.to_csv(
        f"{unique_id_path}_metrics.csv",
        index=False,
    )

    logger.info("###MODEL###")
    logger.info(f"{str(model)}")
    logger.info(f"Job length: {(time.time() - start) / 60} minutes")
    logger.info("###CONFIG###")
    logger.info(f"{yaml.dump(settings, default_flow_style=False)}")
