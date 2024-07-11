from evoVAE.trainer.seq_trainer import (
    calc_covariances,
    plot_and_save_covariances,
)
import pandas as pd
import time, os, argparse
from pathlib import Path

CONFIG_FILE = 1
ARRAY_ID = 2
ALIGN_FILE = -2
MULTIPLE_REPS = 3
PROCESSES = 4
HAS_REPLICATES = 0
SEQ_LEN = 0
BATCH_ZERO = 0
SEQ_ZERO = 0

# errors
SUCCESS = 0
INVALID_FILE = 2


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
        description="Calculate pairwise covarance for reconstructions",
    )

    parser.add_argument(
        "-r",
        "--recon",
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
        "-f",
        "--folds",
        action="store",
        default=5,
        type=int,
        help="Number of k-folds. Defaults to 5 if not specified",
    )

    return parser.parse_args()


def setup_parser() -> argparse.Namespace:
    """use argpase to sort CLI arguments and
    return the args."""

    parser = argparse.ArgumentParser(
        prog="Multiplxed Ancestral Phylogeny (MAP)",
        description="Calc covar",
    )

    parser.add_argument(
        "-re",
        "--reconstruction",
        action="store",
        metavar="example.pkl",
        help="The pkl DataFrame where the reconstructions originate from",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="output.csv.",
        action="store",
        help="output filename. If not \
        specified, a filename called pearson.out will be created in the current working directory.",
    )

    return parser.parse_args()


def validate_file(path):
    """Check that a valid file has been provided,
    otherwise exits with error code INVALID_FILE"""

    if (file := Path(path)).is_file():
        return file

    print(f"{path} is not a valid file. Aborting...")
    exit(INVALID_FILE)


### Main program ###
args = setup_parser()

actual_and_recons = pd.read_pickle(args.reconstruction)

start = time.time()

actual = actual_and_recons[["id", "sequence"]]
recon = actual_and_recons[["id", "reconstruction"]]
recon = recon.rename(columns={"reconstruction": "sequence"})

# find the pairwise covariances of each column in the MSAs
actual_covar, predicted_covar = calc_covariances(
    recon, actual, int(os.getenv("SLURM_CPUS_PER_TASK"))
)

# Calculate correlation coefficient and save an image to file
correlation_coefficient = plot_and_save_covariances(
    actual_covar, predicted_covar, args.output
)

final_metrics = pd.DataFrame({"id": args.output, "pearson": [correlation_coefficient]})
final_metrics.to_csv(args.output + "_pearson.csv", index=False)

print(f"Elapsed minutes: {(time.time() - start) / 60}")
