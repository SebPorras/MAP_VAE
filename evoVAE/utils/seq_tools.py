"""
seq_tools.py
Contains any functions used for general processing of sequences 
before they are passed to the VAE. 
"""

import numpy as np
from typing import Tuple, Dict, List
import pandas as pd
import evoVAE.utils.metrics as mt
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from ete3 import Tree

import torch, re, math, random


GAPPY_PROTEIN_ALPHABET = [
    "-",
    "R",
    "H",
    "K",
    "D",
    "E",
    "S",
    "T",
    "N",
    "Q",
    "C",
    "G",
    "P",
    "A",
    "V",
    "I",
    "L",
    "M",
    "F",
    "Y",
    "W",
]

INVALID_PROTEIN_CHARS = ["B", "J", "X", "Z"]
RE_INVALID_PROTEIN_CHARS = "|".join(map(re.escape, INVALID_PROTEIN_CHARS))

GAPPY_ALPHABET_LEN = len(GAPPY_PROTEIN_ALPHABET)
IDX_TO_AA = dict((idx, acid) for idx, acid in enumerate(GAPPY_PROTEIN_ALPHABET))
AA_TO_IDX = dict((acid, idx) for idx, acid in enumerate(GAPPY_PROTEIN_ALPHABET))


def read_fasta_file(filename: str):

    with open(filename, "r") as file:
        lines = file.readlines()
        sequence = ""
        id = ""
        saveSeq = False
        data = []

        for line in lines:

            if line[0] == ">":
                if saveSeq:

                    data.append([id, sequence])

                id = line[1:].strip()
                saveSeq = True
                sequence = ""

            else:
                sequence += line.strip()

        # add the last sequence
        data.append([id, sequence])

    return data


def read_aln_file(
    filename: str,
    encode: bool = True,
) -> pd.DataFrame:
    """Read in an alignment file in Fasta format and
    return a Pandas DataFrame with sequences and IDs. If encode
    is true, a one-hot encoding will be made."""

    print(f"Reading the alignment: {filename}")

    data = read_fasta_file(filename)
    columns = ["id", "sequence"]
    df = pd.DataFrame(data, columns=columns)

    # handles at a2m file format
    to_upper = lambda x: x.upper().replace(".", "-")
    df["sequence"] = df["sequence"].apply(to_upper)

    # remove sequences with bad characters using regular expressions
    print(f"Checking for bad characters: {INVALID_PROTEIN_CHARS}")
    orig_size = len(df)
    df = df[~df["sequence"].str.contains(RE_INVALID_PROTEIN_CHARS)]
    if orig_size != len(df):
        print(f"Removed {orig_size - len(df)} sequences")

    if encode:
        print("Performing one hot encoding")
        one_hot = df["sequence"].apply(seq_to_one_hot)
        df["encoding"] = one_hot

    print(f"Number of seqs: {len(df)}")
    return df


def parseDefline(string):
    """Parse the FASTA defline (see http://en.wikipedia.org/wiki/FASTA_format)
    GenBank, EMBL, etc                gi|gi-number|gb|accession|locus
    SWISS-PROT, TrEMBL                sp|accession|name
    ...
    Return a tuple with
    [0] primary search key, e.g. UniProt accession, Genbank GI
    [1] secondary search key, e.g. UniProt name, Genbank accession
    [2] source, e.g. 'sp' (SwissProt/UniProt), 'tr' (TrEMBL), 'gb' (Genbank)
    """
    if len(string) == 0:
        return ("", "", "", "")
    s = string.split()[0]
    if re.match("^sp\|[A-Z][A-Z0-9]*\|\S+", s):
        arg = s.split("|")
        return (arg[1], arg[2], arg[0], "")
    elif re.match("^tr\|[A-Z][A-Z0-9]*\|\S+", s):
        arg = s.split("|")
        return (arg[1], arg[2], arg[0], "")
    elif re.match("^gi\|[0-9]*\|\S+\|\S+", s):
        arg = s.split("|")
        return (arg[1], arg[3], arg[0], arg[2])
    elif re.match("gb\|\S+\|\S+", s):
        arg = s.split("|")
        return (arg[1], arg[2], arg[0], "")
    elif re.match("emb\|\S+\|\S+", s):
        arg = s.split("|")
        return (arg[1], arg[2], arg[0], "")
    elif re.match("^refseq\|\S+\|\S+", s):
        arg = s.split("|")
        return (arg[1], arg[2], arg[0], "")
    elif re.match("[A-Z][A-Z0-9]*\|\S+", s):
        arg = s.split("|")
        return (arg[0], arg[1], "UniProt", "")  # assume this is UniProt
    else:
        return (s, "", "", "")


def write_fasta(id: str, seq: str) -> str:
    """Write one sequence in FASTA format to a string and return it."""

    fasta = ">" + id + "\n"
    nlines = int(math.ceil((len(seq) - 1) / 60 + 1))
    for i in range(nlines):
        lineofseq = "".join(seq[i * 60 : (i + 1) * 60]) + "\n"
        fasta += lineofseq
    return fasta


def write_fasta_file(filename: str, data: pd.DataFrame) -> None:
    """Apply string FASTA formatting to a DataFrame and write this
    to a file.
    """

    formatted = data.apply(lambda row: write_fasta(row["id"], row["sequence"]), axis=1)

    with open(filename, "w") as file:
        for seq in formatted:
            file.write(seq)


def seq_to_one_hot(seq: str) -> np.ndarray:
    """Rows are positions in the sequence,
    the columns are the one hot encodings.
    """

    encoding = np.zeros((len(seq), GAPPY_ALPHABET_LEN))

    for column, letter in enumerate(seq):
        encoding[column][AA_TO_IDX[letter]] = 1

    return encoding


def one_hot_to_seq(encoding: torch.Tensor, is_tensor: True) -> str:
    """Take a 2D array with shape (SeqLen, AA_LEN)
    and convert it back to a string."""

    # get the index of the maximum value, which corresponds to a
    # particular character
    if is_tensor:
        encoding = encoding.numpy()

    aa_indices = np.argmax(encoding, axis=1)

    return "".join(IDX_TO_AA[char] for char in aa_indices)


def reweight_sequences(sequences: np.ndarray, theta: float) -> np.ndarray:
    """Take in a Series of sequences and calculate the new weights. Sequences
    are deemed to be clustered if (mutation_count/seq_len) < theta."""

    weights = np.ones(len(sequences))

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if (
                mt.hamming_distance(sequences[i], sequences[j]) / len(sequences[i])
                < theta
            ):
                weights[i] += 1
                weights[j] += 1

    return np.fromiter((map(lambda x: 1.0 / x, weights)), dtype=float)


def encode_and_weight_seqs(
    seqs: pd.Series,
    theta: float,
    reweight=True,
) -> Tuple[np.ndarray, np.ndarray]:

    print("Encoding the sequences and calculating weights")

    # encodings = np.stack(seqs.apply(seq_to_one_hot))
    encodings = seqs.apply(seq_to_one_hot)
    print(f"The sequence encoding has size: {encodings.shape}\n")

    weights = None
    if reweight:
        weights = reweight_sequences(seqs.values, theta=theta)
        print(f"The sequence weight array has size: {weights.shape}\n")

    return encodings, weights


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
    alphabet: List[str] = GAPPY_PROTEIN_ALPHABET,
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

    # position_freq_matrix(pfm)
    pfm = calc_position_freq_matrix(seqs)

    # make a position probability matrix
    for column in pfm:
        total = np.sum(column)
        column /= total

    # makes columns positoins in the sequence
    return pfm.T


def calc_position_freq_matrix(seqs: pd.DataFrame) -> np.ndarray:

    encodings = np.stack(seqs["encoding"].values)

    pfm = np.zeros(encodings.shape[1:])

    for seq in seqs["sequence"]:
        for row, letter in enumerate(seq):
            index = AA_TO_IDX[letter]
            pfm[row][index] += 1

    return pfm


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

    plot_info = [[] for x in range(GAPPY_ALPHABET_LEN)]

    for residues in data.values():
        for aa_index in range(GAPPY_ALPHABET_LEN):
            plot_info[aa_index].append(residues[aa_index])

    # Extract category labels, means, and standard deviations
    plt.figure(figsize=(10, 6))
    plt.xlabel("Residue")
    plt.ylabel("Average Residue Proportion")
    plt.grid(True)
    plt.boxplot(plot_info)
    plt.xticks([i for i in range(1, GAPPY_ALPHABET_LEN + 1)], GAPPY_PROTEIN_ALPHABET)

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


def sample_ancestor_nodes(
    rejection_threshold: float, tree: Tree, tree_run: int
) -> Tuple[List[str], int]:
    """Do a level order traversal of a tree, at each ancestor, only sample if a random number
    between 0 and 1 is below 1 - rejection_threshold.

    Returns:
    list of node names, the number of nodes sampled
    """

    sampled_nodes = []
    for node in tree.traverse():

        # we're only interested in sampling ancestors
        if node.is_leaf():
            continue

        if random.random() <= 1 - rejection_threshold:
            # pkl file format
            sampled_nodes.append(node.name + f"_tree_{tree_run}")

    return sampled_nodes, len(sampled_nodes)


def sample_ancestor_trees(
    tree_count: int,
    rejection_threshold: float,
    invalid_trees: List[int],
    tree_path: str,
    ancestor_pkl_path: str,
    ete_read_setting: int = 1,
):
    """
    Take a path to where all trees are stored. For each tree, sample the ancestor nodes, only taking ancestors and only sampling
    when a random number is less than 1 - rejection_threshold.

    Returns:
    A dataframe with all the sampled_ancestors.
    """

    sampled_names = []
    for iteration in range(1, tree_count + 1):

        if iteration in invalid_trees:
            continue

        # default read setting is 1, to allow internal nodes to be read
        t = Tree(tree_path + f"run_{iteration}_ancestors.nwk", ete_read_setting)
        names, _ = sample_ancestor_nodes(rejection_threshold, t, iteration)
        sampled_names += names

    sampled_ancestors = pd.read_pickle(ancestor_pkl_path)
    sampled_ancestors = sampled_ancestors.loc[
        sampled_ancestors["id"].isin(sampled_names)
    ]

    return sampled_ancestors
