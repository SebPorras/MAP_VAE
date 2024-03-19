"""
seqParser.py
Contains any functions used for general processing of sequences 
before they are passed to the VAE. 
"""

from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple
import pandas as pd
import torch
from datasets import MSA_Dataset

AA = [
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
AA_LEN = len(AA)
IDX_TO_AA = dict((idx, acid) for idx, acid in enumerate(AA))
AA_TO_IDX = dict((acid, idx) for idx, acid in enumerate(AA))


def read_aln_file(
    filename: str,
) -> pd.DataFrame:
    """Read in an alignment file in Fasta format and
    return a Pandas DataFrame with sequences and IDs"""

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


def hamming_distance(seq1: str, seq2: str) -> float:
    """Take two aligned sequences of the same length
    and return the Hamming distance between the two."""

    if len(seq1) != len(seq2):
        raise ValueError("Sequences are the not the same length")

    mutations = 0

    for i, j in zip(seq1, seq2):
        if i != j:
            mutations += 1

    return mutations * 1.0


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
) -> Tuple[torch.tensor, np.ndarray]:

    print("Encoding the sequences and calculating weights")

    encodings = torch.tensor(np.stack(seqs.apply(seq_to_one_hot)), dtype=torch.float32)
    print(f"The sequence encoding tensor has size: {encodings.size()}")

    weights = reweight_sequences(seqs.values, theta=theta)
    print(f"The sequence weight array has size: {weights.shape}\n")

    return encodings, weights


def main():

    THETA = 0.2
    alns: pd.DataFrame = read_aln_file("./data/alignments/tiny.aln")
    train, val = train_test_split(alns, test_size=0.2)

    train_encodings, train_weights = encode_and_weight_seqs(train["sequence"], THETA)
    train_ids = train["id"].values

    train_dataset = MSA_Dataset(train_encodings, train_weights, train_ids)
    en, _, _ = train_dataset[1]
    print(one_hot_to_seq(en))

    val_encodings, val_weights = encode_and_weight_seqs(val["sequence"], THETA)
    val_ids = val["id"].values

    val_dataset = MSA_Dataset(val_encodings, val_weights, val_ids)


if __name__ == "__main__":
    main()
