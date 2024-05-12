from pandas import DataFrame
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.loss.standard_loss import frange_cycle_linear
import evoVAE.utils.metrics as mt
import evoVAE.utils.seq_tools as st
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt


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
    device,
    config: Dict,
) -> SeqVAE:

    model = model.to(device)

    optimiser = model.configure_optimiser(
        learning_rate=config.learning_rate, weight_decay=config.weight_decay
    )

    wandb.define_metric("epoch")
    wandb.define_metric("ELBO", step_metric="epoch")
    wandb.define_metric("KLD", step_metric="epoch")
    wandb.define_metric("Gauss_likelihood", step_metric="epoch")

    wandb.define_metric("val_ELBO", step_metric="epoch")
    wandb.define_metric("val_KLD", step_metric="epoch")
    wandb.define_metric("val_Gauss_likelihood", step_metric="epoch")

    anneal_schedule = frange_cycle_linear(config.epochs)
    early_stopper = EarlyStopper(patience=config.patience)

    for iteration in range(config.epochs):
        train_loop(
            model, train_loader, optimiser, device, config, iteration, anneal_schedule
        )

        stop_early = validation_loop(
            model,
            val_loader,
            device,
            dms_data,
            metadata,
            iteration,
            config,
            anneal_schedule,
            early_stopper,
        )

        if stop_early:
            break

    model.cpu()
    torch.save(model.state_dict(), f"{config.info}_model_state.pt")

    return model


def train_loop(
    model: SeqVAE,
    train_loader: DataLoader,
    optimiser,
    device: str,
    config,
    epoch: int,
    anneal_schedule: np.ndarray,
) -> None:

    epoch_loss = 0
    epoch_kl = 0
    epoch_likelihood = 0
    batch_count = 0

    # ignore seq names for now when training
    model.train()
    for encoding, weights, _ in train_loader:

        encoding = encoding.float().to(device)
        weights = weights.float().to(device)

        # forward step
        optimiser.zero_grad()
        modelOutputs = model(encoding)

        # calculate loss
        loss, kl, likelihood = model.loss_function(
            modelOutputs, encoding, weights, epoch, anneal_schedule
        )
        #print(loss, kl, likelihood)

        # update epoch metrics
        epoch_loss += loss.item()
        epoch_kl += kl.item()
        epoch_likelihood += likelihood.item()
        batch_count += 1

        # update weights
        loss.backward()
        # sets max value for gradient - currently 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimiser.step()

    # log batch results
    wandb.log(
        {
            "ELBO": epoch_loss / batch_count,
            "KLD": epoch_kl / batch_count,
            "Gauss_likelihood": epoch_likelihood / batch_count,
            "epoch": epoch,
        }
    )


def validation_loop(
    model: SeqVAE,
    val_loader: DataLoader,
    device,
    dms_data: DataFrame,
    metadata: DataFrame,
    current_epoch: int,
    config,
    anneal_schedule: np.ndarray,
    early_stopper: EarlyStopper,
) -> bool:
    """
    Calculate loss on validation set. Will also evaluate how well
    the model can predict fitness for unseen variants.

    Returns:
    True if the model should stop early if validation loss is
    increasing, otherwise False.
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

            outputs_val = model(encoding_val)

            loss_val, kl_val, likelihood_val = model.loss_function(
                outputs_val, encoding_val, weight_val, current_epoch, anneal_schedule
            )

            epoch_val_elbo += loss_val.item()
            epoch_val_kl += kl_val.item()
            epoch_val_likelihood += likelihood_val.item()
            batch_count += 1

    wandb.log(
        {
            "val_ELBO": epoch_val_elbo / batch_count,
            "val_KLD": epoch_val_kl / batch_count,
            "val_Gauss_likelihood": epoch_val_likelihood / batch_count,
            "epoch": current_epoch,
        }
    )

    stop_early = early_stopper.early_stop((epoch_val_elbo / batch_count))

    # predict variant fitnesses
    zero_shot_prediction(model, dms_data, metadata, config, current_epoch, stop_early)

    return stop_early


def zero_shot_prediction(
    model: SeqVAE,
    dms_data: pd.DataFrame,
    metadata: pd.DataFrame,
    config,
    current_epoch: int,
    stop_early: bool,
):

    # split variants by how many mutations they have
    subset_dms = split_by_mutations(dms_data)
    for count, subset_mutants in subset_dms.items():

        # Predict fitness of DMS variants with {count} mutations dataset
        if count > config.max_mutation:
            continue

        sub_spear_rho, sub_k_recall, sub_ndcg, sub_roc_auc = fitness_prediction(
            model,
            subset_mutants,
            count,
            metadata,
            current_epoch,
            config.epochs,
            stop_early,
        )
        wandb.log(
            {
                f"{count}_mutations_spearman_rho": sub_spear_rho,
                f"{count}_top_k_recall": sub_k_recall,
                f"{count}_ndcg": sub_ndcg,
                f"{count}_roc_auc": sub_roc_auc,
                "epoch": current_epoch,
            }
        )

    # Predict fitness of DMS variants for ENTIRE dataset
    spear_rho, k_recall, ndcg, roc_auc = fitness_prediction(
        model,
        dms_data,
        None,
        metadata,
        current_epoch,
        config.epochs,
        stop_early,
    )
    wandb.log(
        {
            "spearman_rho": spear_rho,
            "top_k_recall": k_recall,
            "ndcg": ndcg,
            "roc_auc": roc_auc,
            "epoch": current_epoch,
        }
    )


def fitness_prediction(
    model: SeqVAE,
    dms_data: DataFrame,
    mutation_count: int,
    metadata: DataFrame,
    current_epoch: int,
    max_epoch: int,
    stop_early: bool,
) -> Tuple[float, float, float, float]:
    """
    Briefly, the model produces the representation of the
    wild type and the variant to produce a log odds score.

    The model outputs a position probability matrix which can
    estimate the likelihood of a sequence according to the model.
    This is then compared to true fitness values, specifically how
    well the rankings of variants align using a number of metrics.

    log(variant_fitness / wild_type_fitness) = log odds score

    Returns: spear_rho, k_recall, ndcg, roc_auc.
    """

    # encode the wild type
    wild_type = metadata["target_seq"].to_numpy()[0]
    # add dim to the front to allow model to process it
    wild_one_hot = torch.Tensor(st.seq_to_one_hot(wild_type)).unsqueeze(0)

    # get model representation of the wild type
    with torch.no_grad():
        wild_model_encoding, _, _, _ = model(wild_one_hot)

        # reshape into (1, seq_len, AA_count)
        orig_shape = wild_model_encoding.shape[0:-1]
        wild_model_encoding = torch.unsqueeze(wild_model_encoding, -1)
        wild_model_encoding = wild_model_encoding.view(
            orig_shape + (-1, model.AA_COUNT)
        )

        # remove first dim which is just 1 for both tensors
        wild_model_encoding = wild_model_encoding.squeeze(0)
        wild_one_hot = wild_one_hot.squeeze(0)

        # estimate the log ikelihood of sequence based on model output
        wt_prob = mt.seq_log_probability(wild_one_hot, wild_model_encoding)

        # pass all variants through the model
        variant_encodings = torch.Tensor(np.stack(dms_data["encoding"].values))
        variant_model_outputs, _, _, _ = model(variant_encodings)

    # now make fitness estimates
    model_scores = []
    for variant, var_one_hot in zip(variant_model_outputs, variant_encodings):

        # take flat model output and reshape into (seq_len, AA_count)
        var_model_encoding = torch.unsqueeze(variant, -1)
        var_model_encoding = var_model_encoding.view(orig_shape + (-1, model.AA_COUNT))
        var_model_encoding = var_model_encoding.squeeze(0)

        log_prob = mt.seq_log_probability(var_one_hot, var_model_encoding)

        # log(variant_fitness / wild_type_fitness) = log odds score
        model_scores.append(log_prob - wt_prob)

    # compare to ground truth
    model_scores = pd.Series(model_scores)
    spear_rho, k_recall, ndcg, roc_auc = mt.summary_stats(
        predictions=model_scores,
        actual=dms_data["DMS_score"],
        actual_binned=dms_data["DMS_score_bin"],
    )

    # Plot predictions vs actual fitness values but only on final epoch
    # or if early stopping has been triggered.
    if (current_epoch == max_epoch - 1) or stop_early:

        title = "Predicted vs Actual Fitness: "
        if mutation_count is None:
            title += "(All variatns)"
        else:
            title += f"({mutation_count} mutation variants)"
        fig, ax = plt.subplots()

        ax.scatter(dms_data["DMS_score"], model_scores)
        plt.title(title)
        plt.xlabel("Actual")
        plt.ylabel("Prediction")
        wandb.log({title: fig})

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
        subframes[count] = dms_data[dms_data["mut_count"] == count]

    return subframes
