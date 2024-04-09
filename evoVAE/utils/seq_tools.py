"""
seq_tools.py
Contains any functions used for general processing of sequences 
before they are passed to the VAE. 
"""

import numpy as np
from typing import Tuple
import pandas as pd
import torch
import re

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

INVALID_PROTEIN_CHARS = ["X"]
RE_INVALID_PROTEIN_CHARS = "|".join(map(re.escape, INVALID_PROTEIN_CHARS))

AA_LEN = len(GAPPY_PROTEIN_ALPHABET)
IDX_TO_AA = dict((idx, acid) for idx, acid in enumerate(GAPPY_PROTEIN_ALPHABET))
AA_TO_IDX = dict((acid, idx) for idx, acid in enumerate(GAPPY_PROTEIN_ALPHABET))


def read_aln_file(
    filename: str,
    encode: bool = True,
) -> pd.DataFrame:
    """Read in an alignment file in Fasta format and
    return a Pandas DataFrame with sequences and IDs. If encode
    is true, a one-hot encoding will be made."""

    print(f"Reading the alignment: {filename}")

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

    columns = ["id", "sequence"]
    df = pd.DataFrame(data, columns=columns)

    # handles at a2m file format
    to_upper = lambda x: x.upper().replace(".", "-")
    df["sequence"] = df["sequence"].apply(to_upper)

    # remove sequences with bad characters using regular expressions
    print("Checking for bad characters")
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


def seq_to_one_hot(seq: str) -> np.ndarray:
    """Rows are positions in the sequence,
    the columns are the one hot encodings.
    """

    encoding = np.zeros((len(seq), AA_LEN))

    for column, letter in enumerate(seq):
        encoding[column][AA_TO_IDX[letter]] = 1

    return encoding


def one_hot_to_seq(encoding: torch.Tensor) -> str:
    """Take a 2D array with shape (SeqLen, AA_LEN)
    and convert it back to a string."""

    # get the index of the maximum value, which corresponds to a
    # particular character
    aa_indices = np.argmax(encoding.numpy(), axis=1)
    return "".join(IDX_TO_AA[char] for char in aa_indices)


def reweight_sequences(sequences: np.ndarray, theta: float) -> np.ndarray:
    """Take in a Series of sequences and calculate the new weights. Sequences
    are deemed to be clustered if (mutation_count/seq_len) < theta."""

    weights = np.ones(len(sequences))

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if hamming_distance(sequences[i], sequences[j]) / len(sequences[i]) < theta:
                weights[i] += 1
                weights[j] += 1

    return np.fromiter((map(lambda x: 1.0 / x, weights)), dtype=float)


def encode_and_weight_seqs(
    seqs: pd.Series, theta: float
) -> Tuple[np.ndarray, np.ndarray]:

    print("Encoding the sequences and calculating weights")

    # encodings = np.stack(seqs.apply(seq_to_one_hot))
    encodings = seqs.apply(seq_to_one_hot)
    print(f"The sequence encoding has size: {encodings.shape}\n")

    weights = reweight_sequences(seqs.values, theta=theta)
    print(f"The sequence weight array has size: {weights.shape}\n")

    return encodings, weights
