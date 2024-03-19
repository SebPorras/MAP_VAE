"""
seqParser.py
Contains any functions used for general processing of sequences 
before they are passed to the VAE. 
"""

from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch

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

    encoding = np.zeros((AA_LEN, len(seq)))

    for column, letter in enumerate(seq):
        encoding[AA_TO_IDX[letter]][column] = 1

    return encoding


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


def main():

    THETA = 0.2
    alns: pd.DataFrame = read_aln_file("./data/alignments/tiny.aln")
    train, val = train_test_split(alns, test_size=0.2)

    # create one hot embeddings
    train_encodings = np.stack(train["sequence"].apply(seq_to_one_hot))
    # get sequence reweights for this dataset using original seqs
    train_weights = reweight_sequences(train["sequence"].values, theta=THETA)
    print(train_encodings.shape)
    print(train_weights.shape)

    val_encodings = np.stack(val["sequence"].apply(seq_to_one_hot))
    val_weights = reweight_sequences(val["sequence"].values, theta=THETA)
    print(val_encodings.shape)
    print(val_weights.shape)

    indices = torch.tensor([0, 2, 1, 0])

    # Define the number of classes (vocabulary size)
    num_classes = 3  # Example: A classification task with 3 classes

    # Use torch.nn.functional.one_hot to perform one-hot encoding
    one_hot_encoded = F.one_hot(indices, num_classes)

    print(one_hot_encoded)
    print(one_hot_encoded.shape)


if __name__ == "__main__":
    main()
