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
    recon, actual, 8  # int(os.getenv("SLURM_CPUS_PER_TASK"))
)

# Calculate correlation coefficient and save an image to file
correlation_coefficient = plot_and_save_covariances(
    actual_covar, predicted_covar, args.output
)

final_metrics = pd.DataFrame({"id": args.output, "pearson": [correlation_coefficient]})
final_metrics.to_csv(args.output + "_pearson.csv", index=False)

print(f"Elapsed minutes: {(time.time() - start) / 60}")


# import pandas as pd
# import numpy as np
# from evoVAE.trainer.seq_trainer import (
#     calc_covariances,
# )
# CPUS = 10
# path = "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/reconstruction_validation/k_fold_val/"
# protein = ["mafg"]
# data = ["e"]#, "e"]
# labels = ["Extants"]#, "Extants"]

# for protein in protein:
#     for d, label in zip(data, labels):
#         f = open(f"{protein}_{d}_fold_covariances.csv", "w")
#         f.write("unique_id,pearson,category,family\n")
#         for fold in range(1, 6):

#             #file = f"{path}{protein}_{d}_5_fold_r1_fold_{fold}_val_recons.pkl"
#             file = f"{path}{protein}_{d}_r1_fold_{fold}_val_recons.pkl"

#             actual_and_recons = pd.read_pickle(file)
#             actual = actual_and_recons[["id", "sequence"]]
#             recon = actual_and_recons[["id", "reconstruction"]]
#             recon = recon.rename(columns={"reconstruction": "sequence"})

#             id = f"{protein}_{d}_fold_{fold}"

#             actual_covar, predicted_covar = calc_covariances(recon, actual, CPUS)
#             correlation_coefficient = np.corrcoef(actual_covar, predicted_covar)[0, 1]
#             f.write(f"{id},{correlation_coefficient},{label},{protein.upper()}\n")
#             f.flush()

#         f.close()

### CALC HAMMING ###

# import pandas as pd
# import numpy as np
# import evoVAE.utils.seq_tools as st
# import evoVAE.utils.metrics as mt

# CPUS = 10

# path = "/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/reconstruction_validation/k_fold_val/"
# proteins = ["a4"]#, "mafg", "gfp", "gb1", "a4"]
# categories = ["ae", "a", "e"]
# labels = ["Ancestors/Extants", "Ancestors", "Extants"]


# for protein in proteins:
#     f = open(f"{protein}_k_fold_hamming.csv", "w")
#     f.write("unique_id,hamming,category,family\n")
#     for cat, lab in zip(categories, labels):
#         for fold in range(1, 5):

#             file = f"{path}{protein}_{cat}_5_fold_r1_fold_{fold}_val_recons.pkl"

#             actual_and_recons = pd.read_pickle(file)
#             actual = actual_and_recons[["id", "sequence"]]
#             recon = actual_and_recons[["id", "reconstruction"]]
#             recon = recon.rename(columns={"reconstruction": "sequence"})

#             a_msa, _, _ = st.convert_msa_numpy_array(actual)
#             r_msa, _, _ = st.convert_msa_numpy_array(recon)
#             hams = [mt.hamming_distance(a, r)/r_msa.shape[1] for a,r in zip(a_msa, r_msa)]
#             mean_difference = np.mean(hams)

#             id = f"{protein}_{cat}_fold_{fold}"

#             f.write(f"{id},{mean_difference},{lab},{protein.upper()}\n")
#             f.flush()

# f.close()
