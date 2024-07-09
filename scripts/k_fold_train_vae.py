"""k_fold_train_vae.py"""

import torch.utils
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seq_trainer import seq_train
from sklearn.model_selection import KFold
import pandas as pd
import evoVAE.utils.seq_tools as st
from evoVAE.trainer.seq_trainer import sample_latent_space
from datetime import datetime
import yaml, time, os, torch, argparse
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
    # overwrite the alignment in the config file
    if args.aln is not None:
        settings["alignment"] = args.aln

    start = time.time()

    # unique identifier for this experiment
    unique_id_path = Path(args.output + "_r" + args.replicate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Read in the training dataset
    if settings["alignment"].split(".")[-1] in ["fasta", "aln"]:
        aln = st.read_aln_file(settings["alignment"])
    else:
        aln = pd.read_pickle(settings["alignment"])

    # add weights to the sequences
    numpy_aln, _, _ = st.convert_msa_numpy_array(aln)
    weights = st.position_based_seq_weighting(
        numpy_aln, n_processes=int(os.getenv("SLURM_CPUS_PER_TASK"))
    )
    # weights = st.reweight_by_seq_similarity(numpy_aln, theta=0.2)
    aln["weights"] = weights

    # one-hot encode
    one_hot = aln["sequence"].apply(st.seq_to_one_hot)
    aln["encoding"] = one_hot

    train_dataset = MSA_Dataset(
        aln["encoding"].to_numpy(), aln["weights"].to_numpy(), aln["id"], device
    )

    # get the sequence length from first sequence
    seq_len = train_dataset[BATCH_ZERO][SEQ_ZERO].shape[SEQ_LEN]
    input_dims = seq_len * settings["AA_count"]

    log = ""
    log += f"Seq length: {seq_len}\n"

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
        log += f"Fold {fold + 1}"
        log += "-------"

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=settings["batch_size"],
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )

        val_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=settings["batch_size"],
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )

        log += f"Train shape: {len(train_loader)}\n"
        log += f"Validation shape: {len(val_loader)}\n"

        # instantiate the model
        model = SeqVAE(
            dim_latent_vars=settings["latent_dims"],
            dim_msa_vars=input_dims,
            num_hidden_units=settings["hidden_dims"],
            settings=settings,
            num_aa_type=settings["AA_count"],
        )

        # Training Loop
        trained_model = seq_train(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            dms_data=None,
            metadata=None,
            device=device,
            config=settings,
            unique_id=f"{unique_id_path}_fold_{fold + 1}_",
        )

        torch.save(
            trained_model.state_dict(),
            f"{unique_id_path}_fold_{fold + 1}_model_state.pt",
        )

        # Model performance
        num_samples = 100
        test_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,  # will expand to num_samples so need only 1 per batch
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )

        # get the reconstruction of the test set.
        ids, x_hats = sample_latent_space(trained_model, test_loader, num_samples)
        # save for later as we don't need the GPU
        recon = pd.DataFrame({"id": ids, "reconstructions": x_hats})
        recon.to_pickle(f"{unique_id_path}_fold_{fold + 1}_reconstructions.pkl")

    # save config for the run
    yaml_str = yaml.dump(settings, default_flow_style=False)
    with open(f"{unique_id_path}_fold_{fold + 1}_log.txt", "w") as file:
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

    return SUCCESS


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
        "--zero-shot",
        action="store_true",
        help="When specified, zero-shot prediction will be performed. This assumes you have \
            specified a DMS file fitness values.",
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
