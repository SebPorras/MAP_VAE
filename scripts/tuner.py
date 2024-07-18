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
from evoVAE.trainer.seq_trainer import train_loop
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
    original_aln: pd.DataFrame, device: torch.device
) -> MSA_Dataset:

    train_aln = original_aln.copy()
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
        action="store",
        default="output",
    )


    return parser.parse_args()

def objective(trial, aln, device, unique_id_path):

    # get the sequence length from first sequence
    seq_len = len(aln["sequence"][0])
    num_seq = aln.shape[0]
    input_dims = seq_len * 21

    log = ""
    log += f"Seq length: {seq_len}\n"
    log += f"Original aln size : {num_seq}\n"

    # subset the data
    train, val = train_test_split(aln, test_size=0.2, random_state=42)

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

    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
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
        with torch.no_grad():
            # can reuse the recon loader with a single batch size for multiple samples
            elbos = []
            for x, _, _ in val_loader:
                log_elbo = model.compute_elbo_with_multiple_samples(
                    x, num_samples=5000
                )
                elbos.append(log_elbo.item())

        # get average log ELBO for the validation set
        mean_elbo = np.mean(elbos)

        trial.report(mean_elbo, current_epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mean_elbo


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

    train_val, test = train_test_split(aln, test_size=0.2, random_state=42)
    print(f"Train/Val shape: {train_val.shape}")
    print(f"Test shape: {test.shape}")

    f = open(unique_id_path + ".log", "w") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f.write(f"Using device: {device}\n")

    study = optuna.create_study(direction="maximize")
    obj = partial(objective, aln=train_val, device=device, unique_id_path=unique_id_path)
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
