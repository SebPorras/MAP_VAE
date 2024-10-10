"""
tuner.py

Uses Optuna to find hyperparamter settings for the SeqVAE model.
Does this with 5-fold cross validation and the training objective is 
to maximise the marginal probability of the validation set.
"""

import torch.utils
from MAP_VAE.utils.datasets import MSA_Dataset
from MAP_VAE.models.seqVAE import SeqVAE
import pandas as pd
import MAP_VAE.utils.seq_tools as st
import yaml, os, torch, argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from MAP_VAE.loss.standard_loss import frange_cycle_linear
import optuna
from MAP_VAE.trainer.seq_trainer import train_loop
from optuna.trial import TrialState
from functools import partial
import logging
from datetime import datetime


CONFIG_FILE = 1
ARRAY_ID = 2
ALIGN_FILE = -2
MULTIPLE_REPS = 3
SEQ_LEN = 0
BATCH_ZERO = 0
SEQ_ZERO = 0
ALL_VARIANTS = 0
FOLDS = 5

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
        numpy_aln, n_processes=2  # int(os.getenv("SLURM_CPUS_PER_TASK"))
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
        description="Uses Optuna to find hyperparamter settings",
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
        action="store",
        default="output",
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


def objective_cv(trial, aln, device, logger):

    # get the sequence length from first sequence
    seq_len = len(aln["sequence"].values[0])
    num_seq = aln.shape[0]
    input_dims = seq_len * 21

    # work out each index for the k-folds
    num_seq_subset = num_seq // FOLDS + 1
    idx_subset = []
    np.random.seed(42)
    random_idx = np.random.permutation(range(num_seq))
    for i in range(FOLDS):
        idx_subset.append(random_idx[i * num_seq_subset : (i + 1) * num_seq_subset])

    logger.info("-------")
    logger.info(f"Total training size: {num_seq}")
    logger.info(f"Number of folds : {FOLDS}")
    logger.info(f"Fold size: {len(idx_subset[0])}")
    logger.info("-------")

    # hold ids and marginal probability of the validation set
    fold_elbos = []
    for fold in range(FOLDS):
        logger.info(f"Fold {fold + 1}")
        logger.info("-------")

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
            batch_size=1,
        )

        elbo = objective(trial, input_dims, train_loader, val_loader)
        fold_elbos.append(elbo)

    return np.mean(fold_elbos)


def objective(trial, input_dims, train_loader, val_loader):

    latent_dims = trial.suggest_int("latent_dims", 3, 10)
    settings["latent_dims"] = latent_dims

    # instantiate the model
    model = SeqVAE(
        dim_latent_vars=latent_dims,
        dim_msa_vars=input_dims,
        num_hidden_units=settings["hidden_dims"],
        settings=settings,
        num_aa_type=settings["AA_count"],
    )

    model = model.to(device)

    weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-4)
    optimiser = model.configure_optimiser(
        learning_rate=settings["learning_rate"], weight_decay=weight_decay
    )

    scheduler = CosineAnnealingLR(optimiser, T_max=settings["epochs"])
    anneal_schedule = frange_cycle_linear(settings["epochs"])

    for current_epoch in range(settings["epochs"]):

        elbo, recon, kld = train_loop(
            model,
            train_loader,
            optimiser,
            current_epoch,
            anneal_schedule,
            scheduler,
        )

    ### Estimate marginal probability assigned to validation sequences. ###
    model.eval()
    elbos = []
    with torch.no_grad():
        for x, _, _ in val_loader:
            log_elbo = model.compute_elbo_with_multiple_samples(x, num_samples=2000)
            elbos.append(log_elbo.item())

    return np.mean(elbos)


if __name__ == "__main__":

    args = setup_parser()

    # read in the config file
    with open(args.config, "r") as stream:
        settings = yaml.safe_load(stream)

    # overwrite the alignment in the config file
    if args.aln is not None:
        settings["alignment"] = args.aln

    # Read in the training dataset
    if settings["alignment"].split(".")[-1] in ["fasta", "aln"]:
        aln = st.read_aln_file(settings["alignment"])
    else:
        aln = pd.read_pickle(settings["alignment"])

    # unique identifier for this experiment
    unique_id_path = args.output

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

    train_val, test = train_test_split(aln, test_size=0.2, random_state=42)

    logger.info(f"Alignment: {args.aln}")
    logger.info(f"Train/Val shape: {train_val.shape}")
    logger.info(f"Test shape: {test.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}\n")

    study = optuna.create_study(direction="maximize")

    obj = partial(objective_cv, aln=train_val, device=device, logger=logger)

    minutes = 90  # largest models take about 1.5 hours for 5-fold validation
    study.optimize(obj, n_trials=100, timeout=(60 * minutes))

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics: \n")
    logger.info(f"Number of finished trials: {len(study.trials)}\n")
    logger.info(f"Number of pruned trials: {len(pruned_trials)}\n")
    logger.info(f"Number of complete trials: {len(complete_trials)}\n")

    logger.info("Best trial:\n")
    trial = study.best_trial

    logger.info(f"Value: {trial.value}\n")

    logger.info("Params: \n")
    for key, value in trial.params.items():
        logger.info("{}: {}\n".format(key, value))
