# statistics.py

import numpy as np
from typing import Dict, List
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import evoVAE.utils.seq_tools as st
import os
from joblib import dump, load, Parallel, delayed
from numba import njit

COLS = 1
SEQ_COUNT = 0


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


def calc_position_prob_matrix(seqs: pd.DataFrame, pseudo: float = 0.0) -> np.ndarray:
    """
    Take in a DataFrame and create a position probabilty matrix.

    Source:
    https://github.com/loschmidt/vae-dehalogenases

    Returns:
    An array of size (Alphabet_len, seq_len)
    """

    # position_freq_matrix(pfm)
    pfm = calc_position_freq_matrix(seqs, pseudo)

    # normalise
    SEQ_COUNT = 0
    ppm = pfm / seqs.shape[SEQ_COUNT]

    # makes columns positins in the sequence
    return ppm


def calc_position_freq_matrix(seqs: pd.DataFrame, pseudo: float = 0.0) -> np.ndarray:
    """
    Take in a DataFrame and create a position frequence matrix.

    Source:
    https://github.com/loschmidt/vae-dehalogenases

    Returns:
    An array of size (Alphabet_len, seq_len)
    """

    msa, _, _ = st.convert_msa_numpy_array(seqs)

    # shape (21, seq_len)
    pfm = np.zeros((st.GAPPY_ALPHABET_LEN, msa.shape[COLS]))

    for j in range(msa.shape[COLS]):
        col_j = msa[:, j]
        for residue in range(st.GAPPY_ALPHABET_LEN):
            pfm[residue, j] = np.where(col_j == residue)[0].shape[SEQ_COUNT]

    if pseudo > 0:
        pfm += pseudo

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
    """ """

    msa, _, _ = st.convert_msa_numpy_array(seqs)

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


def pair_wise_covariances_parallel(
    msa: np.ndarray, num_processes: int = 2
) -> np.ndarray:
    """
    Takes a MSA and calculates the covariances of amino acids
    across columns. This is the parallel implementation.

    In this implementation, each worker process is given a column to
    work on. This does lead to some imbalances in work done because
    of the upper triangular shape but i don't have time to
    improve this.

    Returns:
    The covariance matrix (np.ndarray): This is a triangular matrix to save memory
    In case you want to find a specific cov score based on column and residue indices in the upper tri matrix
    col_combination_count = (num_cols*(num_cols-1)/2) - (num_cols-col_1_idx)*((num_cols-col_1_idx)-1)/2 + col_2_idx - col_1_idx - 1
    covar_index = int(col_combination_count * aa_combinations + a_idx * st.GAPPY_ALPHABET_LEN + b_idx)
    """

    # number of different residue combinations we can have
    aa_combinations = st.GAPPY_ALPHABET_LEN**2
    num_columns = msa.shape[COLS]

    # the number of unique ways we can compare columns in the MSA
    column_combinations = msa.shape[COLS] * (msa.shape[COLS] - 1) // 2

    # because we are using triangular matrix and are sending columns to each worker
    # we need to keep track of the correct position in the matrix.
    col_combinations = []
    col_combination_count = 0  # starting from first col, each pair has id
    for i in range(num_columns - 1):
        col_combinations.append((i, col_combination_count))
        # add the unique combinations to the count
        col_combination_count += len(range(i + 1, num_columns))

    # access tmp directory for data sharing across processes
    folder = os.environ.get("TMPDIR")

    # dump the MSA onto the file system to speed up process
    msa_filename_memmap = os.path.join(folder, "msa_memmap")
    dump(msa, msa_filename_memmap)
    msa = load(msa_filename_memmap, mmap_mode="r")

    # this will be where our output covariance matrix is dumped
    covariances = np.zeros(column_combinations * aa_combinations)
    output_filename_memmap = os.path.join(folder, "output_memmap")
    output = np.memmap(
        output_filename_memmap,
        dtype=covariances.dtype,
        shape=len(covariances),
        mode="w+",
    )

    # submit columns to each worker
    Parallel(n_jobs=num_processes)(
        delayed(calc_columns_pair_covar)(msa, i, idx, output)
        for i, idx in col_combinations
    )

    # clean up, TMPDIR is removed at end of slurm job.
    os.remove(output_filename_memmap)
    os.remove(msa_filename_memmap)

    return output


@njit()
def calc_columns_pair_covar(
    msa: np.ndarray, i: int, start_idx: int, output: np.ndarray
) -> None:
    """
    Processes a single MSA column and compares it to all other combinations
    of columns. Note that this has also been optimised with Numba.

    Parameters:
    msa: The MSA to process, written to disk with joblib
    i: index of the column to process
    start_idx: where to start in the triangular matrix
    output: covariance matrix, written to disk with joblib
    """

    num_seqs = msa.shape[0]
    num_columns = msa.shape[1]

    col_i = msa[:, i]
    col_combination_count = start_idx
    for j in range(i + 1, num_columns):
        for a in range(st.GAPPY_ALPHABET_LEN):
            for b in range(a + 1, st.GAPPY_ALPHABET_LEN):

                col_j = msa[:, j]
                freq_Ai_Bj, freq_Ai, freq_Bj = calc_col_freqs(
                    col_i, col_j, a, b, num_seqs
                )

                covar_index = (
                    col_combination_count * st.GAPPY_AA_COMBINATIONS
                    + a * st.GAPPY_ALPHABET_LEN
                    + b
                )

                output[covar_index] = freq_Ai_Bj - (freq_Ai * freq_Bj)

        # keep track of how many column combinations we've seen
        col_combination_count += 1


@njit()
def calc_col_freqs(
    col_i: np.ndarray, col_j: np.ndarray, a: int, b: int, num_seqs: int
) -> float:
    """
    Calculates the joint probability of observing residue a
    and b in columns i and j, respectively. Also finds the
    independent frequencies of residue a in column i and residue
    b in column j.

    Returns:
    freq_Ai_Bj: joint probability
    freq_Ai: independent probability
    freq_Bj: independent probability
    """

    # find how many sequences have residues a and b
    col_i_res = np.where(col_i == a)[0]
    col_j_res = np.where(col_j == b)[0]

    # how many times do these residues appear together in these two columns
    intersect = np.intersect1d(col_i_res, col_j_res).shape[SEQ_COUNT]

    # make a frequency based on number of sequences
    freq_Ai_Bj = intersect / num_seqs

    # just count how many sequences have these residues
    freq_Ai = col_i_res.shape[0] / num_seqs
    freq_Bj = col_j_res.shape[0] / num_seqs

    return freq_Ai_Bj, freq_Ai, freq_Bj


def pair_wise_covariances(msa: np.ndarray):
    """
    Takes a MSA and calculates the covariances of amino acids
    across columns.

    Returns:
    The covariance matrix (np.ndarray): This is a triangular matrix to save memory
    In case you want to find a specific cov score based on column and residue indices in the upper tri matrix
    col_combination_count = (num_cols*(num_cols-1)/2) - (num_cols-col_1_idx)*((num_cols-col_1_idx)-1)/2 + col_2_idx - col_1_idx - 1
    covar_index = int(col_combination_count * aa_combinations + a_idx * st.GAPPY_ALPHABET_LEN + b_idx)
    """

    pairs = []
    for i in range(st.GAPPY_ALPHABET_LEN):
        pairs.extend([(i, j) for j in range(i + 1, st.GAPPY_ALPHABET_LEN)])

    # the number of unique ways we can compare columns in the MSA
    column_combinations = msa.shape[COLS] * (msa.shape[COLS] - 1) // 2
    # number of different residue combinations we can have

    num_seqs = msa.shape[SEQ_COUNT]
    num_columns = msa.shape[COLS]

    # each column has aa_combinations many ways to combine residues
    # this is an upper triangular matrix but we will store it in a linear format.
    covariances = np.zeros(column_combinations * st.GAPPY_AA_COMBINATIONS)

    # keep track of which column combination we're up to
    col_combination_count = 0
    for i in range(num_columns - 1):
        for j in range(i + 1, num_columns):
            col_i = msa[:, i]
            col_j = msa[:, j]

            for a, b in pairs:

                # find how many sequences have residues a and b
                col_i_res = np.where(col_i == a)[0]
                col_j_res = np.where(col_j == b)[0]

                # find how many sequences have this combination
                intersect = np.intersect1d(
                    col_i_res, col_j_res, assume_unique=True
                ).shape[SEQ_COUNT]
                # make a frequency based on number of sequences
                freq_Ai_Bj = intersect / num_seqs

                # just count how many sequences have these residues
                freq_Ai = col_i_res.shape[0] / num_seqs
                freq_Bj = col_j_res.shape[0] / num_seqs

                # get correct position: (which column combination we're at) + (which residue combination we're at)
                covar_index = (
                    col_combination_count * st.GAPPY_AA_COMBINATIONS
                    + a * st.GAPPY_ALPHABET_LEN
                    + b
                )

                # useful in case you want to find a specific cov score based on column and residue indices in the upper tri matrix
                # col_combination_count = (num_cols*(num_cols-1)/2) - (num_cols-col_1_idx)*((num_cols-col_1_idx)-1)/2 + col_2_idx - col_1_idx - 1
                # covar_index = int(col_combination_count * aa_combinations + a_idx * st.GAPPY_ALPHABET_LEN + b_idx)

                # (joint occurances of residues a & b at thi) - (occurence of A at col i * occurence of B at col j)
                covariances[covar_index] = freq_Ai_Bj - (freq_Ai * freq_Bj)

            # keep track of how many column combinations we've seen
            col_combination_count += 1

    return covariances


###### VISUALISATION FUNCTIONS #######


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
