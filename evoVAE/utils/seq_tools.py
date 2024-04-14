"""
seq_tools.py
Contains any functions used for general processing of sequences 
before they are passed to the VAE. 
"""

import numpy as np
from typing import Tuple
import pandas as pd
import torch, re, math
import evoVAE.utils.metrics as mt


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
