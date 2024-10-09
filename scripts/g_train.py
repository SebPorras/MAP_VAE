"""g_train.py"""

from sklearn.model_selection import train_test_split
import torch.utils
from src.utils.datasets import MSA_Dataset
from src.models.seqVAE import SeqVAE
from src.trainer.seq_trainer import seq_train
import pandas as pd
import src.utils.seq_tools as st
from datetime import datetime
import yaml, time, os, torch, argparse
from pathlib import Path
import matplotlib.pyplot as plt
import logging

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


def prepare_dataset(original_aln: pd.DataFrame, device: torch.device) -> MSA_Dataset:

    # add weights to the sequences
    numpy_aln, _, _ = st.convert_msa_numpy_array(original_aln)
    weights = st.position_based_seq_weighting(
        numpy_aln, n_processes=int(os.getenv("SLURM_CPUS_PER_TASK"))
    )
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


def setup_parser() -> argparse.Namespace:
    """use argpase to sort CLI arguments and
    return the args."""

    parser = argparse.ArgumentParser(
        prog="Multiplxed Ancestral Phylogeny (MAP)",
        description="Train an instance of a VAE using ancestors",
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
        help="Provide an output name which is adding to the beginning of all files that are created.",
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

    # subset the data
    train, val = train_test_split(
        aln, test_size=settings["test_split"], random_state=42
    )
    logger.info(f"Train/Val shape: {train.shape}")

    logger.info(f"Alignment: {args.aln}")
    logger.info(f"Seq length: {seq_len}")
    logger.info(f"Original aln size : {num_seq}")
    logger.info(f"Test shape: {val.shape}")
    logger.info(f"Using device: {device}")

    # one-hot encodes and weights seqs before sending to device
    train_dataset = prepare_dataset(train, device)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
    )

    val_dataset = prepare_dataset(val, device)
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
        unique_id=unique_id_path,
    )

    # plot the loss for visualtion of learning
    losses = pd.read_csv(f"{unique_id_path}_loss.csv")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    plt.figure(figsize=(12, 8))
    plt.plot(losses["epoch"], losses["elbo"], label="train", marker="o", color="b")
    plt.plot(
        losses["epoch"], losses["val_elbo"], label="validation", marker="x", color="r"
    )
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.legend()
    plt.title(f"{unique_id_path}")
    plt.savefig(f"{unique_id_path}_loss.png", dpi=300)

    torch.save(
        trained_model.state_dict(),
        f"{unique_id_path}_model_state.pt",
    )
    logger.info("Model saved")

    logger.debug(f"Elapsed time: {(time.time() - start) / 60} minutes\n")
    logger.info("###MODEL###")
    logger.info(f"{str(model)}")
    logger.info(f"Job length: {(time.time() - start) / 60} minutes")
    logger.info("###CONFIG###\n")
    logger.info(f"{yaml.dump(settings, default_flow_style=False)}\n")
