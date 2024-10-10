"""
zero_shot.py

Will train a VAE with all available sequences and then predict fitness 
of the variants using only the log odds score between the query sequence 
and the wild-type. 

"""

import torch.utils
from MAP_VAE.utils.datasets import MSA_Dataset
from MAP_VAE.models.seqVAE import SeqVAE
from MAP_VAE.trainer.seq_trainer import fitness_prediction
import pandas as pd
import MAP_VAE.utils.seq_tools as st
from MAP_VAE.trainer.seq_trainer import sample_latent_space
from datetime import datetime
import yaml, time, os, torch, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from MAP_VAE.trainer.seq_trainer import frange_cycle_linear, train_loop

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
        numpy_aln, n_processes=2  # int(os.getenv("SLURM_CPUS_PER_TASK"))
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

    parser.add_argument(
        "-d",
        "--dms",
        action="store",
        help="Deep Mutational Scanning dataset",
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

    if args.dms is not None:
        settings["dms_file"] = args.dms

    # dms_data = pd.read_csv(settings["dms_file"])
    # one_hot = dms_data["mutated_sequence"].apply(st.seq_to_one_hot)
    # dms_data["encoding"] = one_hot

    # metadata = pd.read_csv(settings["dms_metadata"])
    # metadata = metadata[metadata["DMS_id"] == settings["dms_id"]]

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
    logger.info(f"Number of seqs: {num_seq}")
    logger.info(f"Using device: {device}")

    # one-hot encodes and weights seqs before sending to device
    train_dataset = prepare_dataset(aln, device)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
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
        learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=settings["epochs"])
    anneal_schedule = frange_cycle_linear(settings["epochs"])

    with open(unique_id_path + "_loss.csv", "w") as file:
        file.write("epoch,elbo,kld,recon\n")
        file.flush()

        for current_epoch in range(settings["epochs"]):

            elbo, recon, kld = train_loop(
                model,
                train_loader,
                optimiser,
                current_epoch,
                anneal_schedule,
                scheduler,
            )

            file.write(f"{current_epoch},{elbo},{kld},{recon}\n")
            file.flush()

    # plot the loss for visualtion of learning
    # losses = pd.read_csv(f"{unique_id_path}_loss.csv")
    # logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    # plt.figure(figsize=(12, 8))
    # plt.plot(losses["epoch"], losses["elbo"], label="train", marker="o", color="b")
    # plt.xlabel("Epoch")
    # plt.ylabel("ELBO")
    # plt.legend()
    # plt.title(f"{unique_id_path}")
    # plt.savefig(f"{unique_id_path}_loss.png", dpi=300)

    torch.save(
        model.state_dict(),
        f"{unique_id_path}_model_state.pt",
    )
    logger.info("Model saved")
    logger.debug("Starting zero-shot prediction")

    # will also create a plot of actual vs predicted.
    # TODO: Add the code to scale up for different numbers of mutations
    # will hold our metrics
    # spear_rho, k_recall, ndcg, roc_auc = fitness_prediction(
    #     model,
    #     dms_data,
    #     metadata,
    #     f"{unique_id_path}",
    #     device,
    #     mutation_count=ALL_VARIANTS,
    #     n_samples=5000,
    # )

    ### Reconstruction of extant aln ###
    # logger.info("Reconstructing extant alignment\n")
    # if settings["extant_aln"].split(".")[-1] in ["fasta", "aln"]:
    #     extant_aln = st.read_aln_file(settings["extant_aln"])
    # else:
    #     extant_aln = pd.read_pickle(settings["extant_aln"])

    # numpy_aln, _, _ = st.convert_msa_numpy_array(extant_aln)
    # weights = st.position_based_seq_weighting(
    #     numpy_aln, n_processes=int(os.getenv("SLURM_CPUS_PER_TASK"))
    # )
    # weights = st.reweight_by_seq_similarity(numpy_aln, theta=0.2)
    # extant_aln["weights"] = weights
    # extant_aln["encoding"] = extant_aln["sequence"].apply(st.seq_to_one_hot)

    # extant_dataset = MSA_Dataset(
    #     extant_aln["encoding"].to_numpy(),
    #     extant_aln["weights"].to_numpy(),
    #     extant_aln["id"],
    #     device,
    # )

    # extant_loader = torch.utils.data.DataLoader(extant_dataset, batch_size=1, shuffle=False)
    # ids, x_hats = sample_latent_space(
    #     model=model, data_loader=extant_loader, num_samples=100
    # )
    # # save for later as we don't need the GPU
    # recon = pd.DataFrame(
    #     {
    #         "id": ids,
    #         "sequence": extant_aln[extant_aln["id"].isin(ids)]["sequence"],
    #         "reconstruction": x_hats,
    #     }
    # )
    # recon.to_pickle(f"{unique_id_path}_extant_recons.pkl")
    logger.debug(f"Elapsed time: {(time.time() - start) / 60} minutes\n")

    # store our metrics
    # all_metrics = pd.DataFrame(
    #     {
    #         "unique_id": [unique_id_path],
    #         "spearman_rho": [spear_rho],
    #         "top_k_recall": k_recall,
    #         "ndcg": [ndcg],
    #         "roc_auc": [roc_auc]

    #     }
    # )

    # all_metrics.to_csv(
    #     f"{unique_id_path}_metrics.csv",
    #     index=False,
    # )

    logger.info("###MODEL###")
    logger.info(f"{str(model)}")
    logger.info(f"Job length: {(time.time() - start) / 60} minutes")
    logger.info("###CONFIG###\n")
    logger.info(f"{yaml.dump(settings, default_flow_style=False)}\n")
