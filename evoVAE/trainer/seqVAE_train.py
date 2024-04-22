from pandas import DataFrame
from evoVAE.models.seqVAE import SeqVAE
import evoVAE.utils.metrics as mt
import evoVAE.utils.seq_tools as st
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple
from torch.utils.data import DataLoader
import wandb


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

    for iteration in range(config.epochs):
        train_loop(model, train_loader, optimiser, device, config)
        validation_loop(model, val_loader, device, dms_data, metadata)

    model.cpu()
    torch.save(model, "seqVAE_weights.pt")

    return model


def train_loop(
    model: SeqVAE, train_loader: DataLoader, optimiser, device, config
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
        loss, kl, likelihood = model.loss_function(modelOutputs, encoding)

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
            "epoch_ELBO": epoch_loss / batch_count,
            "epoch_KLD": epoch_kl / batch_count,
            "epoch_Gauss_likelihood": epoch_likelihood / batch_count,
        }
    )


def validation_loop(
    model: SeqVAE,
    val_loader: DataLoader,
    device,
    dms_data: DataFrame,
    metadata: DataFrame,
) -> None:

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
                outputs_val, encoding_val
            )

            epoch_val_elbo += loss_val.item()
            epoch_val_kl += kl_val.item()
            epoch_val_likelihood += likelihood_val.item()
            batch_count += 1

    # Predict fitness of DMS variants
    spear_rho, k_recall, ndcg, roc_auc = fitness_prediction(model, dms_data, metadata)

    wandb.log(
        {
            "epoch_val_ELBO": epoch_val_elbo / batch_count,
            "epoch_val_KLD": epoch_val_kl / batch_count,
            "epoch_val_Gauss_likelihood": epoch_val_likelihood / batch_count,
            "spearman_rho": spear_rho,
            "top_k_recall": k_recall,
            "ndcg": ndcg,
            "roc_auc": roc_auc,
        }
    )


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


def fitness_prediction(
    model: SeqVAE,
    dms_data: DataFrame,
    metadata: DataFrame,
) -> Tuple[float, float, float, float]:
    """
    Returns: spear_rho, k_recall, ndcg, roc_auc
    """

    # encode the wild type
    wild_type = metadata["target_seq"].to_numpy()[0]
    # add dim to the front to allow model to process it
    wild_one_hot = torch.Tensor(st.seq_to_one_hot(wild_type)).unsqueeze(0)

    # get model representation of the wild type
    wild_model_encoding, _, _, _ = model(wild_one_hot)

    # reshape into (1, seq_len, AA_count)
    orig_shape = wild_model_encoding.shape[0:-1]
    wild_model_encoding = torch.unsqueeze(wild_model_encoding, -1)
    wild_model_encoding = wild_model_encoding.view(orig_shape + (-1, model.AA_COUNT))

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

        # log(variant_fitness / wild_type_fitness) - log odds score
        model_scores.append(log_prob - wt_prob)

    # compare to ground truth
    model_scores = pd.Series(model_scores)
    spear_rho, k_recall, ndcg, roc_auc = mt.summary_stats(
        predictions=model_scores,
        actual=dms_data["DMS_score"],
        actual_binned=dms_data["DMS_score_bin"],
    )

    return spear_rho, k_recall, ndcg, roc_auc
