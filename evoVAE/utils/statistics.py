# statistics.py

import numpy as np
from typing import Dict, List
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import evoVAE.utils.seq_tools as st


def calc_mean_seq_embeddings(seqs: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Read in an alignment or aln file and calculate the distribution
    of amino acids in each sequence for every sequence in the file.

    Return:
    A dictionary mapping seq ID to embedding means.
    """

    aa_means = {}

    for name, seq in zip(seqs["id"], seqs["encoding"]):
        aa_means[name] = seq.mean(axis=0)

    return aa_means


def calc_average_residue_distribution(
    mean_seq_embeddings: Dict[str, np.ndarray],
    alphabet: List[str] = st.GAPPY_PROTEIN_ALPHABET,
):
    """Calculate the proportion of amino acids in this group of
    sequences but this is column invariant."""

    encodings = np.stack(list(mean_seq_embeddings.values()))
    encoding_mean = encodings.mean(axis=0)
    encoding_std = np.std(encodings, axis=0)

    averages = {
        letter: {"mean": mean, "std": std}
        for letter, mean, std in zip(alphabet, encoding_mean, encoding_std)
    }

    return averages


def calc_position_prob_matrix(seqs: pd.DataFrame):
    """
    Take in a DataFrame and create a position probabilty matrix.

    Source:
    https://github.com/loschmidt/vae-dehalogenases

    Returns:
    An array of size (Alphabet_len, seq_len)
    """

    # position_freq_matrix(pfm)
    pfm = calc_position_freq_matrix(seqs)

    # normalise
    SEQ_COUNT = 0
    ppm = pfm / seqs.shape[SEQ_COUNT]

    # makes columns positoins in the sequence
    return ppm


def calc_position_freq_matrix(seqs: pd.DataFrame) -> np.ndarray:
    """
    Take in a DataFrame and create a position frequence matrix.

    Source:
    https://github.com/loschmidt/vae-dehalogenases

    Returns:
    An array of size (Alphabet_len, seq_len)
    """

    msa, _, _ = st.convert_msa_numpy_array(seqs)

    SEQ_COUNT = 0
    COLS = 1

    # shape (21, seq_len)
    pfm = np.zeros((st.GAPPY_ALPHABET_LEN, msa.shape[COLS]))

    for j in range(msa.shape[COLS]):
        col_j = msa[:, j]
        for residue in range(st.GAPPY_ALPHABET_LEN):
            pfm[residue, j] = np.where(col_j == residue)[0].shape[SEQ_COUNT]

    return pfm


def safe_log(x, eps=1e-10):
    """
    Calculate numerically stable log.

    Source:
    https://github.com/loschmidt/vae-dehalogenases/
    """

    # return -10 if x is less than eps
    result = np.where(x > eps, x, -10)

    # save the result, avoiding zeros or negatives
    np.log(result, out=result, where=result > 0)
    return result


def calc_shannon_entropy(seqs: pd.DataFrame) -> np.ndarray:

    msa, _, _ = st.convert_msa_numpy_array(seqs)

    SEQ_COUNT = 0
    COLS = 1
    # find entropy for each column
    entropy = np.zeros(msa.shape[COLS])

    # shape (21, seq_len)
    pfm = np.zeros((st.GAPPY_ALPHABET_LEN, msa.shape[COLS]))

    for j in range(msa.shape[COLS]):
        col_j = msa[:, j]
        for residue in range(st.GAPPY_ALPHABET_LEN):
            pfm[residue, j] = np.where(col_j == residue)[0].shape[SEQ_COUNT]

        col_prob = pfm[:, j] / seqs.shape[SEQ_COUNT]

        # -SUM(p(x) * log(p(x)))
        entropy[j] = -np.sum(col_prob * safe_log(col_prob))

    return entropy


def create_euclidean_dist_matrix(
    embedding_means: Dict[str, np.ndarray], plot: bool = False
) -> pd.DataFrame:
    """
    Takes a dictionary mapping of seq ID to the distribution
    of amino acids in a sequence and calculates the euclidean
    distance between pairs of sequence distributions.

    Return:
    A DataFrame of the euclidan matrix.
    """

    ids = list(embedding_means.keys())
    embeddings = np.array(list(embedding_means.values()))

    dist_mat = distance_matrix(embeddings, embeddings)

    if plot:
        plt.imshow(dist_mat, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Distance")
        plt.title("Euclidean distance matrix")
        plt.xticks(range(len(embeddings)), ids, rotation=90)
        plt.yticks(range(len(embeddings)), ids)
        plt.xlabel("Samples")
        plt.ylabel("Samples")
        plt.show()

    return pd.DataFrame(dist_mat, index=ids, columns=ids)


def plot_residue_distributions(data: Dict[str, np.ndarray]) -> None:

    plot_info = [[] for x in range(st.GAPPY_ALPHABET_LEN)]

    for residues in data.values():
        for aa_index in range(st.GAPPY_ALPHABET_LEN):
            plot_info[aa_index].append(residues[aa_index])

    # Extract category labels, means, and standard deviations
    plt.figure(figsize=(10, 6))
    plt.xlabel("Residue")
    plt.ylabel("Average Residue Proportion")
    plt.grid(True)
    plt.boxplot(plot_info)
    plt.xticks(
        [i for i in range(1, st.GAPPY_ALPHABET_LEN + 1)], st.GAPPY_PROTEIN_ALPHABET
    )

    plt.show()


def population_profile_deviation(
    population: pd.DataFrame, sample: pd.DataFrame
) -> float:

    # get the average residue proportion across the population
    pop_means = calc_mean_seq_embeddings(population)
    pop_means = calc_average_residue_distribution(pop_means)

    # put this in an array to allow comparisons
    pop_vector = np.array([x["mean"] for x in pop_means.values()])

    sample_means = calc_mean_seq_embeddings(sample)
    sample_n = len(sample_means)

    total_dist = 0.0
    for sample_vector in sample_means.values():
        total_dist += euclidean(pop_vector, sample_vector)

    mean_deviation = total_dist / sample_n

    return mean_deviation
