"""tuner.py"""

import torch.utils
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAE import SeqVAE
import pandas as pd
import evoVAE.utils.seq_tools as st
import yaml, os, torch, argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from evoVAE.loss.standard_loss import frange_cycle_linear
import optuna
from optuna.trial import TrialState
from functools import partial


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


def prepare_dataset(aln: pd.DataFrame, device: torch.device) -> MSA_Dataset:

    # add weights to the sequences
    numpy_aln, _, _ = st.convert_msa_numpy_array(aln)
    weights = st.position_based_seq_weighting(
        numpy_aln, n_processes=int(os.getenv("SLURM_CPUS_PER_TASK"))
    )
    # weights = st.reweight_by_seq_similarity(numpy_aln, theta=0.2)
    aln["weights"] = weights

    # one-hot encode
    aln["encoding"] = aln["sequence"].apply(st.seq_to_one_hot)

    train_dataset = MSA_Dataset(
        aln["encoding"].to_numpy(),
        aln["weights"].to_numpy(),
        aln["id"],
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
        description="Optimise parameters with Optuna",
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
        "-t",
        "--test-split",
        default=0.2,
        type=float,
        action="store",
        help="Test split. Defaults to 0.2",
    )

    return parser.parse_args()


def objective(trial, aln, device, args):

    # get the sequence length from first sequence
    seq_len = len(aln["sequence"][0])
    num_seq = aln.shape[0]
    input_dims = seq_len * 21

    log = ""
    log += f"Seq length: {seq_len}\n"
    log += f"Original aln size : {num_seq}\n"

    # subset the data
    train, val = train_test_split(aln, test_size=args.test_split)

    # one-hot encodes and weights seqs before sending to device
    train_dataset = prepare_dataset(train, device)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
    )

    val_dataset = prepare_dataset(val, device)
    # use batch size of 1 because we will estimate marginal with many samples
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
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

    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    optimiser = model.configure_optimiser(
        learning_rate=settings["learning_rate"], weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=settings["epochs"])
    anneal_schedule = frange_cycle_linear(settings["epochs"])

    for current_epoch in range(settings["epochs"]):

        epoch_loss = 0
        epoch_kl = 0
        epoch_log_PxGz = 0
        batch_count = 0

        # TRAINING
        model.train()
        for encoding, weights, _ in train_loader:

            optimiser.zero_grad()

            elbo, log_PxGz, kld = model.compute_weighted_elbo(
                encoding, weights, anneal_schedule, current_epoch
            )

            # allows for gradient descent
            elbo = (-1) * elbo

            # update epoch metrics
            epoch_loss += elbo.item()
            epoch_kl += kld.item()
            epoch_log_PxGz += log_PxGz.item()
            batch_count += 1

            # update weights
            elbo.backward()
            optimiser.step()

        scheduler.step()  # adjust learning rate
        epoch_loss /= batch_count
        epoch_kl /= batch_count
        epoch_log_PxGz /= batch_count

        # VALIDATION
        model.eval()
        elbos = []
        with torch.no_grad():
            for x, _, _ in val_loader:
                log_elbo = model.compute_elbo_with_multiple_samples(x, num_samples=5000)
                elbos.append(log_elbo.item())

        # get average log ELBO for the validation set
        mean_val_elbo = np.mean(elbos)
        trial.report(mean_val_elbo, current_epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mean_val_elbo


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

    f = open(args.output, "w") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f.write(f"Using device: {device}\n")

    study = optuna.create_study(direction="maximize")
    obj = partial(objective, aln=aln, device=device, args=args)
    study.optimize(obj, n_trials=100, timeout=1200)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    f.write("Study statistics: \n")
    f.write(f"Number of finished trials: {len(study.trials)}\n")
    f.write(f"Number of pruned trials: {len(pruned_trials)}\n")
    f.write(f"Number of complete trials: {len(complete_trials)}\n")

    f.write("Best trial:\n")
    trial = study.best_trial

    f.write(f"Value: {trial.value}\n")

    f.write("Params: \n")
    for key, value in trial.params.items():
        f.write("{}: {}\n".format(key, value))

    f.close()
