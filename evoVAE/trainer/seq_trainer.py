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
from pandas import DataFrame
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# constants
ALL_VARIANTS = 0

# import optuna
# from optuna.trial import TrialState


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
    dms_data: DataFrame,
    metadata: DataFrame,
    device: torch.device,
    config: Dict,
    unique_id: str,
) -> SeqVAE:

    model = model.to(device)

    optimiser = model.configure_optimiser(
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=config["epochs"])

    anneal_schedule = frange_cycle_linear(config["epochs"])
    early_stopper = EarlyStopper(patience=config["patience"])

    with open(unique_id + "loss.csv", "w") as file:
        file.write("epoch,elbo,kld,recon,val_elbo,val_kld,val_recon\n")
        file.flush()

        for iteration in range(config["epochs"]):

            elbo, recon, kld = train_loop(
                model,
                train_loader,
                optimiser,
                device,
                iteration,
                anneal_schedule,
                scheduler,
            )

            stop_early, val_elbo, val_kld, val_recon = validation_loop(
                model,
                val_loader,
                device,
                dms_data,
                metadata,
                iteration,
                config,
                anneal_schedule,
                early_stopper,
                unique_id,
            )

            file.write(
                f"{iteration},{elbo},{kld},{recon},{val_elbo},{val_kld},{val_recon}\n"
            )
            file.flush()

            if stop_early:
                break

    return model


def train_loop(
    model: SeqVAE,
    train_loader: DataLoader,
    optimiser,
    device: str,
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

        encoding = encoding.float().to(device)
        weights = weights.float().to(device)

        # forward step
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
        # sets max value for gradient
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
    device,
    dms_data: DataFrame,
    metadata: DataFrame,
    current_epoch: int,
    config: Dict,
    anneal_schedule: np.ndarray,
    early_stopper: EarlyStopper,
    unique_id: str,
) -> Tuple[bool, float, float, float]:
    """
    Calculate loss on validation set. Will also evaluate how well
    the model can predict fitness for unseen variants.

    Returns:
    early_stop: True if the model should stop early if validation loss is
    increasing, otherwise False.

    val_elbo, val_kld, val_reconstruction_error
    """

    epoch_val_elbo = 0
    epoch_val_kl = 0
    epoch_val_likelihood = 0
    batch_count = 0

    model.eval()
    with torch.no_grad():
        for encoding_val, weight_val, _ in val_loader:

            # Calculate ELBO on validation set
            encoding_val = encoding_val.float().to(device)
            weight_val = weight_val.float().to(device)

            elbo, log_PxGz, kld = model.compute_weighted_elbo(
                encoding_val, weight_val, anneal_schedule, current_epoch
            )

            # allows for gradient descent
            elbo = (-1) * elbo

            epoch_val_elbo += elbo.item()
            epoch_val_kl += kld.item()
            epoch_val_likelihood += log_PxGz.item()
            batch_count += 1

    #stop_early = early_stopper.early_stop((epoch_val_elbo / batch_count))
    stop_early = False

    if config["zero_shot"] and (current_epoch == (config["epochs"] - 1) or stop_early):
        # predict variant fitnesses
        zero_shot_prediction(
            model, dms_data, metadata, config, current_epoch, unique_id, device
        )

    # early_stop, val_elbo, val_kld, val_reconstruction_error
    return (
        stop_early,
        epoch_val_elbo / batch_count,
        epoch_val_likelihood / batch_count,
        epoch_val_kl / batch_count,
    )


def zero_shot_prediction(
    model: SeqVAE,
    dms_data: pd.DataFrame,
    metadata: pd.DataFrame,
    config: Dict,
    current_epoch: int,
    unique_id: str,
    device,
):
    """
    Split the DMS dataset up into subsets based on how many mutations
    each variant has. Measure performance metrics on model predictions
    on these subsets. Then get metrics for performance on the entire
    dataset.

    Returns:
    None, all metrics are logged with WandB.
    """

    # split variants by how many mutations they have
    """
    subset_dms = split_by_mutations(dms_data)
    for count, subset_mutants in subset_dms.items():

        # Predict fitness of DMS variants with {count} mutations dataset
        if count > config["max_mutation"]:
            continue

        sub_spear_rho, sub_k_recall, sub_ndcg, sub_roc_auc = fitness_prediction(
            model,
            subset_mutants,
            count,
            metadata,
            unique_id,
            device
        )
    """


    # Predict fitness of DMS variants for ENTIRE dataset
    spear_rho, k_recall, ndcg, roc_auc = fitness_prediction(
        model, dms_data, metadata, unique_id, device, mutation_count=ALL_VARIANTS
    )


def fitness_prediction(
    model: SeqVAE,
    dms_data: DataFrame,
    metadata: DataFrame,
    unique_id: str,
    device: torch.device,
    mutation_count: int = 0,
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
    )

    dms_loader = torch.utils.data.DataLoader(dms_dataset, batch_size=1, shuffle=False)

    # encode the wild type
    wild_type = metadata["target_seq"].to_numpy()[0]
    # add dim to the front to allow model to process it
    wild_one_hot = torch.Tensor(st.seq_to_one_hot(wild_type)).unsqueeze(0).float()
    wild_one_hot = wild_one_hot.to(device)

    model.eval()
    actual_fitness = []
    actual_fitness_binned = []
    predicted_fitness = []
    ids = []
    with torch.no_grad():

        wt_elbo_mean = model.compute_elbo_with_multiple_samples(wild_one_hot, n_samples)

        for variant_encoding, variant_id, score, score_bin in dms_loader:

            variant_encoding = variant_encoding.float().to(device)

            variant_elbo_mean = model.compute_elbo_with_multiple_samples(
                variant_encoding, n_samples
            )

            pred_fitness = variant_elbo_mean - wt_elbo_mean

            predicted_fitness.append(pred_fitness.item())
            actual_fitness.append(score.item())
            actual_fitness_binned.append(score_bin.item())
            ids.append(variant_id[0])

    # compare to ground truth
    predicted_fitness = pd.Series(predicted_fitness)
    actual_fitness = pd.Series(actual_fitness)
    actual_fitness_binned = pd.Series(actual_fitness_binned)

    spear_rho, k_recall, ndcg, roc_auc = mt.summary_stats(
        predictions=predicted_fitness,
        actual=actual_fitness,
        actual_binned=actual_fitness_binned,
    )

    # save the final metrics to file.
    final_metrics = pd.DataFrame(
        {
            "unique_id": [unique_id],
            "spearman_rho": [spear_rho],
            "top_k_recall": [k_recall],
            "ndcg": [ndcg],
            "roc_auc": [roc_auc],
        }
    )

    # construct a plot of all the predictions
    title = "zero_shot"
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
    final_metrics.to_csv(unique_id + title + "_final_metrics.csv", index=False)

    ax.scatter(actual_fitness, predicted_fitness)
    plt.title(title + "_" + unique_id[2:-1])
    plt.xlabel("Actual")
    plt.ylabel("Prediction")
    plt.savefig(unique_id + title + ".png")

    return spear_rho, k_recall, ndcg, roc_auc


def split_by_mutations(dms_data: DataFrame) -> Dict[int, DataFrame]:
    """
    Create a subset of the mutation DataFrames based on how many mutations
    are in the variant.

    Return:
    A dictionary mapping mutation count to subset dataframe
    """

    # define a function for counting mutations
    splitter = lambda x: len(x.split(":"))
    dms_data["mut_count"] = dms_data["mutant"].apply(splitter)

    subframes = dict()
    for count in dms_data["mut_count"].unique():
        # subframes[count] = dms_data[dms_data["mut_count"] == count]
        subframes[count] = dms_data[dms_data["mut_count"] == count].reset_index(
            drop=True
        )

    return subframes


def calc_reconstruction_accuracy(
    model: SeqVAE,
    aln: pd.DataFrame,
    outfile: str,
    device,
    num_samples: int = 100,
    num_processes: int = 2,
) -> float:

    train_dataset = MSA_Dataset(aln["encoding"], aln["weights"], aln["id"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False
    )

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
    model: SeqVAE, data_loader: MSA_Dataset, device, num_samples: int
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
            x = x.to(device)
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
    ids: List[str],
    reconstructions: List[Tensor],
    aln: pd.DataFrame,
    outfile: str,
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

    recons_df = pd.DataFrame({"id": ids, "sequence": reconstructions})
    recon_msa, _, _ = st.convert_msa_numpy_array(recons_df)
    predicted_covar = stats.pair_wise_covariances_parallel(recon_msa, num_processes)

    # save reconstruction vs actual for visualisation with MSA later
    recons_df["sequence"] = aln["sequence"]
    recons_df["reconstructions"] = reconstructions
    recons_df.to_pickle(outfile + "recon_seqs.pkl")

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

    plt.savefig(outfile + "covar.png")

    return correlation_coefficient
