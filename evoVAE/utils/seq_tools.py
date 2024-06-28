"""
seq_tools.py
Contains any functions used for general processing of sequences 
before they are passed to the VAE. 
"""

import numpy as np
from typing import Tuple, List
import pandas as pd
import evoVAE.utils.metrics as mt
import torch, re, math
from numba import njit, prange
from joblib import Parallel, delayed
import random

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

INVALID_PROTEIN_CHARS = ["B", "J", "X", "Z", "U"]
RE_INVALID_PROTEIN_CHARS = "|".join(map(re.escape, INVALID_PROTEIN_CHARS))

GAPPY_ALPHABET_LEN = len(GAPPY_PROTEIN_ALPHABET)
GAPPY_AA_COMBINATIONS = GAPPY_ALPHABET_LEN**2

IDX_TO_AA = dict((idx, acid) for idx, acid in enumerate(GAPPY_PROTEIN_ALPHABET))
AA_TO_IDX = dict((acid, idx) for idx, acid in enumerate(GAPPY_PROTEIN_ALPHABET))
NUM_SEQS = 0
SEQ_LEN = 1


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
    encode: bool = False,
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


def encode_and_weight_seqs(
    aln: pd.DataFrame,
    theta: float,
    reweight=True,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Return:
    encodings, weights
    """

    print("Encoding the sequences and calculating weights")

    # encodings = np.stack(seqs.apply(seq_to_one_hot))
    encodings = aln["sequence"].apply(seq_to_one_hot)
    print(f"The sequence encoding has size: {encodings.shape}\n")

    msa, _, _ = convert_msa_numpy_array(aln)

    weights = None
    if reweight:
        weights = reweight_by_seq_similarity(msa, theta=theta)
        print(f"The sequence weight array has size: {weights.shape}\n")

    return encodings, weights


def convert_msa_numpy_array(aln: pd.DataFrame) -> Tuple[np.ndarray, List, List]:
    """
    Turn a group of aligned sequences into a numerical format to leverage
    functions in the numpy library.

    Returns:
    seq_msa, seq_key, seq_label
    """
    sequence_pattern_dict = {}
    seq_msa = []
    seq_key = []
    seq_label = []

    lb = 0

    for id, seq in zip(aln["id"], aln["sequence"]):
        seq_trns = [AA_TO_IDX[s] for s in seq]
        seq_trns_m = "".join([str(x) for x in seq_trns])
        seq_msa.append(seq_trns)
        seq_key.append(id)

        if seq_trns_m not in sequence_pattern_dict:
            sequence_pattern_dict.update({seq_trns_m: lb})
            lb = lb + 1

        seq_label.append(sequence_pattern_dict[seq_trns_m])

    seq_msa = np.array(seq_msa)

    print(
        "Sequence weight numpy array created with shape (num_seqs, columns): ",
        seq_msa.shape,
    )
    return seq_msa, seq_label, seq_key


####### SEQUENCE REWEIGHTING #######


@njit(parallel=True)
def reweight_by_seq_similarity(sequences: np.ndarray, theta: float) -> np.ndarray:
    """Take in a Series of sequences and calculate the new weights. Sequences
    are deemed to be clustered if (mutation_count/seq_len) < theta.

    Sequences must be converted into a numerical format using convert_msa_numpy_array()
    or parallelisation will not work.
    """

    weights = np.ones(sequences.shape[0])

    for i in prange(sequences.shape[0]):
        for j in prange(i + 1, sequences.shape[0]):

            if (
                mt.hamming_distance(sequences[i], sequences[j]) / len(sequences[i])
                < theta
            ):
                weights[i] += 1
                weights[j] += 1

    for i in prange(weights.shape[0]):
        weights[i] = 1.0 / weights[i]

    return weights


def position_based_seq_weighting(
    seq_msa: np.ndarray, n_processes: int = 2
) -> np.ndarray:
    """
    Alternative way of reweighting based off
    https://www.nature.com/articles/s41467-019-13633-0#Sec9 and implemented
    originally by Sanjana Tule.

    Uses a mix of jobib process spawning and Numba parallelisation to
    get a speed-up.

    Returns:
    Normalised sequence weights.
    """

    # hand each process an individual column to work on
    chunks = [seq_msa[:, col] for col in range(seq_msa.shape[SEQ_LEN])]
    results = Parallel(n_jobs=n_processes)(
        delayed(process_column)(chunk) for chunk in chunks
    )

    # collect back futures in order they were submitted
    seq_weight = np.column_stack([np.array(r) for r in results])

    # normalise the weights
    tot_weight = np.sum(seq_weight)
    seq_weight = seq_weight.sum(axis=1) / tot_weight

    # print(
    #     "Sequence weight numpy array created with shape (num_seqs, columns): ",
    #     seq_weight.shape,
    # )

    return seq_weight


def process_column(seq_msa_chunk: np.ndarray) -> np.ndarray:
    """

    Identifies unique amino acids within a column and then sends
    work to a Numba compiled function to actually calculate weights.

    Alternative way of reweighting based off
    https://www.nature.com/articles/s41467-019-13633-0#Sec9 and implemented
    originally by Sanjana Tule.

    Returns:
    Raw sequence weights before normalisation.
    """

    aa_type, aa_counts = np.unique(seq_msa_chunk, return_counts=True, axis=0)
    weights = reweight_by_col_freqs(seq_msa_chunk, aa_type, aa_counts)

    return weights


@njit()
def reweight_by_col_freqs(
    seq_msa_chunk: np.ndarray, aa_type: np.ndarray, aa_counts: np.ndarray
) -> np.ndarray:
    """
    Perform the actual reweighting by column frequencies here. Kept
    as a separate function to allow compilation with Numba.

    Alternative way of reweighting based off
    https://www.nature.com/articles/s41467-019-13633-0#Sec9 and implemented
    originally by Sanjana Tule.

    Returns:
    Raw sequence weights before normalisation.
    """

    weights = np.zeros(seq_msa_chunk.shape[0])

    num_type = len(aa_type)
    aa_dict = {}
    for a, count in zip(aa_type, aa_counts):
        aa_dict[a] = count

    # product of how much the column varies and how many time a residue appears
    for i in range(seq_msa_chunk.shape[NUM_SEQS]):
        weights[i] = (1.0 / num_type) * (1.0 / aa_dict[seq_msa_chunk[i]])

    return weights


def sample_clusters(
    clusters: List[pd.DataFrame],
    sample_ids: set,
    clusters_seen: int,
    num_clusters: int,
    cluster_obs: dict,
    is_ancestor: int,
    current_size: int,
) -> Tuple[int, int]:

    cluster_idx = clusters_seen % num_clusters
    current_cluster = clusters[cluster_idx]
    current_cluster = current_cluster[current_cluster["is_ancestor"] == is_ancestor]

    # can't sample if there's no extants in this cluster
    if (
        current_cluster.shape[0] == 0
        or cluster_obs[cluster_idx] == current_cluster.shape[0]
    ):
        clusters_seen += 1
        return current_size, clusters_seen

    sample_idx = random.randint(0, current_cluster.shape[0] - 1)
    sample = current_cluster.iloc[sample_idx, 1:]

    while True:

        if sample["sequence"] not in sample_ids:
            sample_ids.add(sample["sequence"])
            current_size += 1
            clusters_seen += 1
            cluster_obs[cluster_idx] += 1
            break

        else:
            sample_idx = random.randint(0, current_cluster.shape[0] - 1)
            sample = current_cluster.iloc[sample_idx, 1:]

    return current_size, clusters_seen


def sample_extant_ancestors(
    clusters: List[pd.DataFrame], sample_size: int, extant_proportion: float
):

    num_clusters = len(clusters)
    cluster_an_obs = {i: 0 for i in range(num_clusters)}
    cluster_ex_obs = {i: 0 for i in range(num_clusters)}

    current_size = 0
    clusters_seen = 0

    EXTANT = 0
    ANCESTOR = 1

    sample_ids = set()

    while current_size < sample_size:

        while (current_size / sample_size) < extant_proportion:
            current_size, clusters_seen = sample_clusters(
                clusters,
                sample_ids,
                clusters_seen,
                num_clusters,
                cluster_ex_obs,
                EXTANT,
                current_size,
            )

        current_size, clusters_seen = sample_clusters(
            clusters,
            sample_ids,
            clusters_seen,
            num_clusters,
            cluster_an_obs,
            ANCESTOR,
            current_size,
        )

    return sample_ids
