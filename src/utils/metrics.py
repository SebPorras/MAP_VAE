"""
metrics.py

Contains any functions used to calculate statistics 
or performance metrics.
"""

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import src.utils.metrics as mt
from typing import Union, Tuple
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from numba import prange, njit


def summary_stats(
    predictions: pd.Series,
    actual: pd.Series,
    actual_binned: pd.Series,
    k_top: int = 10,
) -> Tuple[float, float, float, float]:
    """
    Inputs:
    actual: an array of the true scores where higher score is better
    predicted: an array of the predicted scores where higher score is better
    top_k: This is a PERCENTAGE (i.e input 10 for top 10%)

    Returns:
    spearmans_rank_correlation, top_k_recall, normalised_discounted_cumulative_gain,
    roc_auc_score,

    Based off the ProteinGym codebase by Notin et al., 2023.
    """

    k_recall = top_k_recall(predictions, actual, k_top)
    ndcg = calc_ndcg(actual, predictions, k_top)

    # handle the case where there is only one class due to mutation counts
    if len(actual_binned.unique()) == 1:
        roc_auc = None
    else:
        roc_auc = roc_auc_score(actual_binned, predictions)

    spear_rho, p_value = spearmanr(predictions, actual)

    return spear_rho, k_recall, ndcg, roc_auc


def top_k_recall(
    predictions: pd.Series, actual: pd.Series, k_pred: int = 10, k_actual: int = 10
) -> float:
    """
    Assumes that predictions and actual values come from a dataframe where mutants have
    been sorted in decending order based on prediction values.
    The 100 - top_kth percentile is calculated and used as a filter
    for both model and prediction values. Recal is defined as TP / (TP + FN).

    Based off the ProteinGym codebase by Notin et al., 2023.
    """

    top_k_pred = predictions >= np.percentile(predictions, 100 - k_pred)
    top_k_actual = actual >= np.percentile(actual, 100 - k_actual)

    TP = top_k_pred & top_k_actual

    return TP.sum() / top_k_actual.sum()


def calc_ndcg(
    actual: Union[pd.Series, np.ndarray],
    predicted: Union[pd.Series, np.ndarray],
    top_k: int = 10,
) -> float:
    """
    Normalised discounted culmulative gains (NDCG).

    Inputs:
        actual: an array of the true scores where higher score is better
        predicted: an array of the predicted scores where higher score is better
        top_k: This is a PERCENTAGE (i.e input 10 for top 10%)

    Based off Notin et al., 2023 - ProteinGym codebase
    """

    if isinstance(actual, pd.Series):
        actual = actual.values
    if isinstance(predicted, pd.Series):
        predicted = predicted.values

    gains = mt.minmax(actual)
    # find indices in decscending order and start counting from 1
    ranks = np.argsort(np.argsort(-predicted)) + 1

    # sub to top k
    ranks_k = ranks[ranks <= top_k]
    gains_k = gains[ranks <= top_k]

    # all terms with a gain of 0 go to 0
    ranks_fil = ranks_k[gains_k != 0]
    gains_fil = gains_k[gains_k != 0]

    # if none of the ranks made it return 0
    if len(ranks_fil) == 0:
        return 0.0

    # discounted cumulative gains
    dcg = np.sum([g / np.log2(r + 1) for r, g in zip(ranks_fil, gains_fil)])

    # ideal dcg - calculated based on the top k actual gains
    ideal_ranks = np.argsort(np.argsort(-gains)) + 1
    ideal_ranks_k = ideal_ranks[ideal_ranks <= top_k]
    ideal_gains_k = gains[ideal_ranks <= top_k]
    ideal_ranks_fil = ideal_ranks_k[ideal_gains_k != 0]
    ideal_gains_fil = ideal_gains_k[ideal_gains_k != 0]
    idcg = np.sum(
        [g / np.log2(r + 1) for r, g in zip(ideal_ranks_fil, ideal_gains_fil)]
    )

    # normalize
    ndcg = dcg / idcg

    return ndcg


def minmax(x: np.ndarray) -> np.ndarray:
    """
    Min-max normalisation
    """

    return (x - np.min(x)) / (np.max(x) - np.min(x))


def seq_log_probability(one_hot: Tensor, model_encoding: Tensor) -> float:
    """Estimate the likelihood of observing a particular sequence using a
    position weight matrix (pwm). Multiply together and then take the trace of the
    matrix.
    """

    product = torch.matmul(model_encoding, one_hot.T)
    # log_product = torch.log(product)
    return torch.trace(product).item()


@njit()
def hamming_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Take two aligned sequences of the same length
    and return the Hamming distance between the two."""

    if len(seq1) != len(seq2):
        raise ValueError("Sequences are the not the same length")

    mutations = 0

    for i in prange(seq1.shape[0]):
        if seq1[i] != seq2[i]:
            mutations += 1

    return mutations
