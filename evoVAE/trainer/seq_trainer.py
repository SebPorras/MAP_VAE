"""seq_trainer.py"""

# Package modules
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.loss.standard_loss import frange_cycle_linear
import evoVAE.utils.metrics as mt
import evoVAE.utils.seq_tools as st
from evoVAE.utils.datasets import MSA_Dataset
import evoVAE.utils.statistics as stats
from evoVAE.utils.datasets import DMS_Dataset

# External libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ALL_VARIANTS = 0


### EARLY STOPPING ###
class EarlyStopper:
    """
    Will trigger when validation loss increases
    """

    def __init__(self, patience=3) -> None:

        self.patience = patience
        self.counter = 0
        self.min_val_loss = None

    def early_stop(self, val_loss: float):

        if self.min_val_loss is None:
            self.min_val_loss = val_loss

        if val_loss > self.min_val_loss:
            self.counter += 1

            if self.counter >= self.patience:
                return True
        else:
            self.counter = 0

        self.min_val_loss = val_loss

        return False


### TRAINING SCRIPTS ###
def seq_train(
    model: SeqVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    unique_id: str,
) -> SeqVAE:

    optimiser = model.configure_optimiser(
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    scheduler = CosineAnnealingLR(optimiser, T_max=config["epochs"])

    anneal_schedule = frange_cycle_linear(config["epochs"])
    early_stopper = EarlyStopper(patience=config["patience"])

    with open(unique_id + "_loss.csv", "w") as file:
        file.write("epoch,elbo,kld,recon,val_elbo,val_kld,val_recon\n")
        file.flush()

        for current_epoch in range(config["epochs"]):

            elbo, recon, kld = train_loop(
                model,
                train_loader,
                optimiser,
                current_epoch,
                anneal_schedule,
                scheduler,
            )

            stop_early, val_elbo, val_kld, val_recon = validation_loop(
                model,
                val_loader,
                current_epoch,
                anneal_schedule,
                early_stopper,
            )

            file.write(
                f"{current_epoch},{elbo},{kld},{recon},{val_elbo},{val_kld},{val_recon}\n"
            )
            file.flush()

            if stop_early:
                break

    return model


def train_loop(
    model: SeqVAE,
    train_loader: DataLoader,
    optimiser,
    epoch: int,
    anneal_schedule: np.ndarray,
    scheduler,
) -> Tuple[float, float, float]:
    """

    Returns:
    elbo, kld, reconstruction_error
    """

    epoch_loss = 0
    epoch_kl = 0
    epoch_log_PxGz = 0
    batch_count = 0

    # ignore seq names for now when training
    model.train()
    for encoding, weights, _ in train_loader:

        optimiser.zero_grad()

        elbo, log_PxGz, kld = model.compute_weighted_elbo(
            encoding, weights, anneal_schedule, epoch
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

    return (
        epoch_loss / batch_count,
        epoch_log_PxGz / batch_count,
        epoch_kl / batch_count,
    )


def validation_loop(
    model: SeqVAE,
    val_loader: DataLoader,
    current_epoch: int,
    anneal_schedule: np.ndarray,
    early_stopper: EarlyStopper,
) -> Tuple[bool, float, float, float]:
    """
    Calculate loss on validation set. Will also evaluate how well
    the model can predict fitness for unseen variants.

    Returns:
    early_stop: True if the model should stop early if validation loss is
    increasing, otherwise False.
    val_elbo,
    val_likelihood,
    val_kl 
    """

    epoch_val_elbo = 0
    epoch_val_kl = 0
    epoch_val_likelihood = 0
    batch_count = 0

    model.eval()
    with torch.no_grad():
        for encoding_val, weight_val, _ in val_loader:

            # Calculate ELBO on validation set
            elbo, log_PxGz, kld = model.compute_weighted_elbo(
                encoding_val, weight_val, anneal_schedule, current_epoch
            )

            # allows for gradient descent
            elbo = (-1) * elbo

            epoch_val_elbo += elbo.item()
            epoch_val_kl += kld.item()
            epoch_val_likelihood += log_PxGz.item()
            batch_count += 1

    # stop_early = early_stopper.early_stop((epoch_val_elbo / batch_count))
    stop_early = False

    return (
        stop_early,
        epoch_val_elbo / batch_count,
        epoch_val_likelihood / batch_count,
        epoch_val_kl / batch_count,
    )


def fitness_prediction(
    model: SeqVAE,
    dms_data: pd.DataFrame,
    metadata: pd.DataFrame,
    unique_id: str,
    device: torch.device,
    mutation_count: int = ALL_VARIANTS,
    n_samples: int = 500,
) -> Tuple[float, float, float, float]:
    """
    Briefly, the model produces the representation of the
    wild type and the variant to produce a log odds score.

    The model outputs a position probability matrix which can
    estimate the likelihood of a sequence according to the model.
    This is then compared to true fitness values, specifically how
    well the rankings of variants align using a number of metrics.

    log(variant_fitness / wild_type_fitness) = log odds score

    Returns:
    spear_rho, k_recall, ndcg, roc_auc.
    """

    dms_dataset = DMS_Dataset(
        dms_data["encoding"],
        dms_data["mutant"],
        dms_data["DMS_score"],
        dms_data["DMS_score_bin"],
        device,
    )

    # because we will do n_samples per observation, batch size is 1.
    # TODO investigate a way to allow batching even with multiple samples
    dms_loader = torch.utils.data.DataLoader(dms_dataset, batch_size=1, shuffle=False)

    # encode the wild type
    wild_type = metadata["target_seq"].to_numpy()[0]
    # add dim to the front to allow model to process it
    wild_one_hot = torch.Tensor(st.seq_to_one_hot(wild_type)).unsqueeze(0)
    wild_one_hot = wild_one_hot.float().to(device)

    model.eval()
    actual_fitness = []
    actual_fitness_binned = []
    predicted_fitness = []
    ids = []
    with torch.no_grad():

        wt_elbo_mean = model.compute_elbo_with_multiple_samples(wild_one_hot, n_samples)

        for variant_encoding, variant_id, score, score_bin in dms_loader:

            variant_elbo_mean = model.compute_elbo_with_multiple_samples(
                variant_encoding, n_samples
            )

            # do a log likelihood ratio against the wild type.
            pred_fitness = variant_elbo_mean - wt_elbo_mean

            predicted_fitness.append(pred_fitness.item())
            actual_fitness.append(score.item())
            actual_fitness_binned.append(score_bin.item())
            ids.append(variant_id[0])

    # compare to ground truth
    predicted_fitness = pd.Series(predicted_fitness)
    actual_fitness = pd.Series(actual_fitness)
    actual_fitness_binned = pd.Series(actual_fitness_binned)

    # calculate all the metrics we are interested in
    spear_rho, k_recall, ndcg, roc_auc = mt.summary_stats(
        predictions=predicted_fitness,
        actual=actual_fitness,
        actual_binned=actual_fitness_binned,
    )

    # construct a plot of all the predictions
    title = "_zero_shot"
    if mutation_count == ALL_VARIANTS:
        title += "_all_variants"
    else:
        title += f"_{mutation_count}_mutations"
    fig, ax = plt.subplots()

    raw_data = pd.DataFrame(
        {
            "mutant": ids,
            "actual": actual_fitness.values,
            "predicted": predicted_fitness.values,
        }
    )

    raw_data.to_csv(unique_id + title + ".csv", index=False)

    ax.scatter(actual_fitness, predicted_fitness)
    plt.title(title + "_" + unique_id[2:-1])
    plt.xlabel("Actual")
    plt.ylabel("Prediction")
    plt.savefig(unique_id + title + ".png")

    return spear_rho, k_recall, ndcg, roc_auc


def calc_reconstruction_accuracy(
    model: SeqVAE,
    aln: pd.DataFrame,
    dataset: MSA_Dataset,
    outfile: str,
    device: torch.device,
    num_samples: int = 100,
    num_processes: int = 2,
) -> float:

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # sample the latent space and get an average reconstruction for each seq
    ids, x_hats = sample_latent_space(model, train_loader, device, num_samples)

    # find the pairwise covariances of each column in the MSAs
    actual_covar, predicted_covar = calc_covariances(
        ids, x_hats, aln, outfile, num_processes
    )

    # Calculate correlation coefficient and save an image to file
    correlation_coefficient = plot_and_save_covariances(
        actual_covar, predicted_covar, outfile
    )

    return correlation_coefficient


def sample_latent_space(
    model: SeqVAE, data_loader: MSA_Dataset, num_samples: int
) -> Tuple[List[str], List[Tensor]]:
    """
    Take a trained model and sample the latent space num_samples many times. This gets the average
    latent space representation of that sequence.

    Returns:
    ids: The ID of each sequence.
    x_hats: The model output for each sequence.
    """

    ids = []

    x_hats = []
    model.eval()
    with torch.no_grad():
        for x, _, id in data_loader:

            # get into flat format to pass through the model
            x = x.expand(num_samples, -1, -1)
            x = torch.flatten(x, start_dim=1)

            # get encoding and replicate to allow multiple samples from latent space
            z_mu, z_sigma = model.encoder(x)
            eps = torch.randn_like(z_sigma)
            z_samples = z_mu + z_sigma * eps

            mean_z = torch.mean(z_samples, dim=0)
            log_p = model.decoder(mean_z)

            fixed_shape = tuple(log_p.shape[0:-1])

            # decode the Z sample and get it into a PPM shape
            x_hat = torch.unsqueeze(log_p, -1)
            x_hat = x_hat.view(fixed_shape + (-1, st.GAPPY_ALPHABET_LEN))

            # Identify most likely residue at each column
            indices = x_hat.max(dim=-1).indices.tolist()
            recon = "".join([st.GAPPY_PROTEIN_ALPHABET[x] for x in indices])

            x_hats.append(recon)
            ids.append(id[0])

    return ids, x_hats


def calc_covariances(
    reconstructions: pd.DataFrame,
    aln: pd.DataFrame,
    num_processes: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes reconstructed sequences and computes the pairwise covariances of
    columns in those sequences. Then does the same thing for the actual MSA.

    Also writes the reconstruction and actual sequences to a pkl file
    so that they can be easily visualised with a MSA later.

    Returns (Tuple[np.ndarray, np.ndarray]):
    actual_covariances, predicted_covariances
    """

    recon_msa, _, _ = st.convert_msa_numpy_array(reconstructions)
    predicted_covar = stats.pair_wise_covariances_parallel(recon_msa, num_processes)

    msa, _, _ = st.convert_msa_numpy_array(aln)
    actual_covar = stats.pair_wise_covariances_parallel(msa, num_processes)

    return actual_covar, predicted_covar


def plot_and_save_covariances(
    actual_covar: np.ndarray, predicted_covar: np.ndarray, outfile: str
) -> float:
    """
    Plot the covariance values of the reconstruction MSA and actual MSA
    on a single and calculate Pearson's correlation.

    Writes a .png to file for reference.

    Returns (float):
    correlation_coefficient
    """

    fig, ax = plt.subplots()

    plt.scatter(actual_covar, predicted_covar)
    plt.xlabel("Extant MSA covariance")
    plt.ylabel("Extant MSA reconstruction covariance")

    # Calculate correlation coefficient, use [0,1] to not take from diagnonal
    correlation_coefficient = np.corrcoef(actual_covar, predicted_covar)[0, 1]

    # Display correlation value
    plt.text(
        plt.xlim()[0],
        plt.ylim()[1],
        f"Pearson's correlation: {correlation_coefficient:.3f}",
        va="top",
    )
    # remove directory '/' and './'
    plt.title(outfile[2:-1])

    plt.savefig(outfile + "_covar.png")

    return correlation_coefficient


# def zero_shot_prediction(
#     model: SeqVAE,
#     dms_data: pd.DataFrame,
#     metadata: pd.DataFrame,
#     config: Dict,
#     current_epoch: int,
#     unique_id: str,
#     device,
# ):
#     """
#     Split the DMS dataset up into subsets based on how many mutations
#     each variant has. Measure performance metrics on model predictions
#     on these subsets. Then get metrics for performance on the entire
#     dataset.

#     Returns:
#     None, all metrics are logged with WandB.
#     """

#     # split variants by how many mutations they have
#     """
#     subset_dms = split_by_mutations(dms_data)
#     for count, subset_mutants in subset_dms.items():

#         # Predict fitness of DMS variants with {count} mutations dataset
#         if count > config["max_mutation"]:
#             continue

#         sub_spear_rho, sub_k_recall, sub_ndcg, sub_roc_auc = fitness_prediction(
#             model,
#             subset_mutants,
#             count,
#             metadata,
#             unique_id,
#             device
#         )
#     """

#     # Predict fitness of DMS variants for ENTIRE dataset
#     spear_rho, k_recall, ndcg, roc_auc = fitness_prediction(
#         model, dms_data, metadata, unique_id, device, mutation_count=ALL_VARIANTS
#     )


# def split_by_mutations(dms_data: DataFrame) -> Dict[int, DataFrame]:
#     """
#     Create a subset of the mutation DataFrames based on how many mutations
#     are in the variant.

#     Return:
#     A dictionary mapping mutation count to subset dataframe
#     """

#     # define a function for counting mutations
#     splitter = lambda x: len(x.split(":"))
#     dms_data["mut_count"] = dms_data["mutant"].apply(splitter)

#     subframes = dict()
#     for count in dms_data["mut_count"].unique():
#         # subframes[count] = dms_data[dms_data["mut_count"] == count]
#         subframes[count] = dms_data[dms_data["mut_count"] == count].reset_index(
#             drop=True
#         )

#     return subframes
