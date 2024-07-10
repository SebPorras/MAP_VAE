from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seq_trainer import (
    calc_covariances,
    plot_and_save_covariances,
)
import pandas as pd
import evoVAE.utils.seq_tools as st
import yaml, time, os, torch, argparse, sys
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


def main():

    args = setup_parser()

    # read in the config file
    with open(args.config, "r") as stream:
        settings = yaml.safe_load(stream)

    unique_id_path = Path(f"{args.output}_r{args.replicate}_fold_{args.folds}")
    recons = pd.read_pickle(f"{unique_id_path}_reconstructions.pkl")

    if args.aln is not None:
        settings["alignment"] = args.aln

    # Read in the training dataset
    if settings["alignment"].split(".")[-1] in ["fasta", "aln"]:
        aln = st.read_aln_file(settings["alignment"])
    else:
        aln = pd.read_pickle(settings["alignment"])

    aln = aln[aln["id"].isin(recons["id"])]

    start = time.time()
    # find the pairwise covariances of each column in the MSAs
    actual_covar, predicted_covar = calc_covariances(
        recons, aln, int(os.getenv("SLURM_CPUS_PER_TASK"))
    )

    # Calculate correlation coefficient and save an image to file
    correlation_coefficient = plot_and_save_covariances(
        actual_covar, predicted_covar, unique_id_path
    )

    final_metrics = pd.DataFrame({"pearson": correlation_coefficient})
    final_metrics.to_csv(
        unique_id_path + "_zero_shot_all_variants_final_metrics.csv", index=False
    )

    print(f"elapsed minutes: {(time.time() - start) / 60}")


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
        help="The alignment where the reconstructions originate from",
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

    return parser.parse_args()


if __name__ == "__main__":
    main()
