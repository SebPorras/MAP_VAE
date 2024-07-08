# %%
from numpy import require
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seq_trainer import seq_train
from sklearn.model_selection import train_test_split
import pandas as pd
import evoVAE.utils.seq_tools as st

import wandb
import sys, yaml, time, os, torch, argparse
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path


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


def main() -> int:

    args = setup_parser()

    # read in the config file
    with open(args.config, "r") as stream:
        settings = yaml.safe_load(stream)

    # If flag is included, zero shot prediction will occur
    settings["zero_shot"] = args.zero_shot

    wandb.login()

    start = time.time()

    # create the output directory
    unique_id_path = Path(args.output + "_r" + args.replicate)
    unique_id_path.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=settings["project"],
        # hyperparameters
        config=settings,
        name=str(unique_id_path),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Read in the training dataset
    if settings["alignment"].split(".")[-1] in ["fasta", "aln"]:
        ancestors_extants_aln = st.read_aln_file(settings["alignment"])
    else:
        ancestors_extants_aln = pd.read_pickle(settings["alignment"])

    # if a replicate file has been provided, subset the data using indices in the file
    if settings["replicate_csv"] is not None:
        replicate_data = pd.read_csv(settings["replicate_csv"])
        # subset based on random sample
        indices = replicate_data["rep_" + args.replicate]
        ancestors_extants_aln = ancestors_extants_aln.loc[indices]

    # add weights to the sequences
    numpy_aln, _, _ = st.convert_msa_numpy_array(ancestors_extants_aln)
    weights = st.position_based_seq_weighting(
        numpy_aln, n_processes=int(os.getenv("SLURM_CPUS_PER_TASK"))
    )
    # weights = st.reweight_by_seq_similarity(numpy_aln, theta=0.2)
    ancestors_extants_aln["weights"] = weights

    # one-hot encode
    one_hot = ancestors_extants_aln["sequence"].apply(st.seq_to_one_hot)
    ancestors_extants_aln["encoding"] = one_hot

    # subset the data
    train, val = train_test_split(
        ancestors_extants_aln, test_size=settings["test_split"]
    )

    # training/validation
    train_dataset = MSA_Dataset(train["encoding"], train["weights"], train["id"])
    val_dataset = MSA_Dataset(val["encoding"], val["weights"], val["id"])

    # DATA LOADERS
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=settings["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=settings["batch_size"], shuffle=True
    )

    # Load and subset the DMS data used for fitness prediction
    dms_data = pd.read_csv(settings["dms_file"])
    one_hot = dms_data["mutated_sequence"].apply(st.seq_to_one_hot)
    dms_data["encoding"] = one_hot

    # grab metadata for current experiment
    metadata = pd.read_csv(settings["dms_metadata"])
    metadata = metadata[metadata["DMS_id"] == settings["dms_id"]]

    # get the sequence length from first sequence
    seq_len = train_dataset[BATCH_ZERO][SEQ_ZERO].shape[SEQ_LEN]
    input_dims = seq_len * settings["AA_count"]

    log = ""
    log += f"Train shape: {train.shape}\n"
    log += f"Validation shape: {val.shape}\n"
    log += f"Seq length: {seq_len}\n"

    # instantiate the model
    model = SeqVAE(
        dim_latent_vars=settings["latent_dims"],
        dim_msa_vars=input_dims,
        num_hidden_units=settings["hidden_dims"],
        settings=settings,
        num_aa_type=settings["AA_count"],
    )

    # #### Training Loop
    trained_model = seq_train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        dms_data=dms_data,
        metadata=metadata,
        device=device,
        config=settings,
        unique_id=str(unique_id_path) + "/",
    )

    torch.save(
        trained_model.state_dict(),
        unique_id_path / f"{unique_id_path}_model_state.pt",
    )

    # plot the loss for visualtion of learning
    losses = pd.read_csv(unique_id_path / "loss.csv")

    plt.figure()
    plt.plot(losses["epoch"], losses["elbo"], label="train", marker="o", color="b")
    plt.plot(
        losses["epoch"], losses["val_elbo"], label="validation", marker="x", color="r"
    )
    plt.xlabel("epoch")
    plt.ylabel("ELBO")
    plt.legend()
    plt.title(unique_id_path)
    plt.savefig(unique_id_path / "loss.png", dpi=300)

    # save config for the run
    yaml_str = yaml.dump(settings, default_flow_style=False)
    with open(unique_id_path / "log.txt", "w") as file:
        file.write(f"run_id: {unique_id_path}\n")
        file.write(f"time: {datetime.now()}\n")
        file.write("###CONFIG###\n")
        file.write(f"{yaml_str}\n")
        file.write("###RUN INFO###\n")
        file.write(log)
        file.write("###MODEL###\n")
        file.write(f"{str(model)}\n")
        file.write("###TIME###\n")
        file.write(f"{(time.time() - start) / 60} minutes\n")

    wandb.finish()

    return SUCCESS


def validate_file(path):
    """Check that a valid file has been provided,
    otherwise exits with error code INVALID_FILE"""

    if (file := Path(path)).is_file():
        return file

    print(f"{path} is not a valid file. Aborting...")
    exit(INVALID_FILE)


def setup_parser() -> argparse.Namespace:
    """use the stdlib argpase to sort CLI arguments and
    return the args."""

    parser = argparse.ArgumentParser(
        prog="Multiplxed ancestral phylogeny (MAP)",
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
        "--zero-shot",
        action="store_true",
        help="When specified, zero-shot prediction will be performed. This assumes you have \
            specified a DMS file fitness values.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
