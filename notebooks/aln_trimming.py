# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import MAP_VAE.utils.seq_tools as st
from MAP_VAE.utils.datasets import MSA_Dataset
import MAP_VAE.utils.statistics as stats
import pandas as pd
import torch
import numpy as np
import concurrent.futures
import threading

from scipy import stats
from Bio import Align, AlignIO


pd.set_option("display.max_rows", None)


# +


def trim_aln(aln: Align.MultipleSeqAlignment, cols):
    """Trim indices in cols from a multiple sequence alignment."""

    # Check that all col indices are in range of sequence length
    if not isinstance(cols, list):
        cols = [cols]
    cols = list(set(cols))  # Remove duplicate column indices
    for col in cols:
        if not 0 <= col < aln.get_alignment_length():
            raise RuntimeError(f"Invalid column index: {col}")

    # Trim from highest to lowest index
    cols.sort()
    for col in cols[::-1]:
        if col == aln.get_alignment_length() - 1:  # Trimming last col of current aln
            aln = aln[:, :-1]
        elif col == 0:  # Trimming first column
            aln = aln[:, 1:]
        else:  # Trimming non-end column and need to combine two sub-alns
            aln = aln[:, :col] + aln[:, col + 1 :]

    return aln


def smart_trim(
    in_file,
    out_file=None,
    format="fasta",
    base_threshold=0.05,
    p_threshold=0.05,
    max_threads=10,
):
    """Trim columns informed by column occupancy and pairwise sequence similarity of sequences with characters in
    low-occupancy positions. Trimming in this manner is likely to remove erroneously aligned columns without a large
    loss of phylogenetic signal, and reduce complexity for indel resolution in ASR. Note that if ASR is being performed
    using the resulting alignment, an assessment should be made as to whether trimmed columns are likely to be present
    in any targetted node."""

    full_aln = AlignIO.read(in_file, format)

    pw_coverage_full = {}

    # Multi-thread pairwise coverage calculations
    pw_coverage_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_threads
    )
    pw_coverage_lock = threading.Lock()

    # Method for pairwise coverage calculation of sequence j with all sequences from j+1 to len(aln)
    def pw_coverage(j):
        # Calculate pairwise alignment coverages (as aligned proportion of shorter sequence)
        for k in range(j + 1, len(full_aln)):
            seqs = {full_aln[j].name: full_aln[j], full_aln[k].name: full_aln[k]}
            # Determine shortest ungapped seq
            shorter = min(
                seqs.keys(), key=lambda seq: len(str(seqs[seq].seq).replace("-", ""))
            )
            longer = [key for key in seqs.keys() if key != shorter][0]
            # Find proportion of shorter seq positions aligned w/ longer
            aligned_cnt = (
                0  # Positions with content in short seq with content also in long seq
            )
            for pos in range(full_aln.get_alignment_length()):
                if seqs[shorter].seq[pos] != "-" and seqs[longer].seq[pos] != "-":
                    aligned_cnt += 1
            aligned_proportion = aligned_cnt / len(
                str(seqs[shorter].seq).replace("-", "")
            )
            with pw_coverage_lock:
                pw_coverage_full[tuple(seqs.keys())] = aligned_proportion
        print(f"PW coverages done for seq {j}")

    future_store = [
        pw_coverage_executor.submit(pw_coverage, j) for j in range(len(full_aln) - 1)
    ]
    concurrent.futures.wait(future_store)

    # TODO: Now in nested function for multi-threading - remove this if it works
    # # Calculate pairwise alignment coverage (as aligned proportion of shorter sequence) for all seqs in full_aln
    # for j in range(len(full_aln) - 1):
    #     for k in range(j + 1, len(full_aln)):
    #         seqs = {full_aln[j].name: full_aln[j], full_aln[k].name: full_aln[k]}
    #         # Determine shortest ungapped seq
    #         shorter = min(seqs.keys(), key=lambda seq: len(str(seqs[seq].seq).replace('-', '')))
    #         longer = [key for key in seqs.keys() if key != shorter][0]
    #         # Find proportion of shorter seq positions aligned w/ longer
    #         aligned_cnt = 0  # Positions with content in short seq with content also in long seq
    #         for pos in range(full_aln.get_alignment_length()):
    #             if seqs[shorter].seq[pos] != '-' and seqs[longer].seq[pos] != '-':
    #                 aligned_cnt += 1
    #         aligned_proportion = aligned_cnt / len(str(seqs[shorter].seq).replace('-', ''))
    #         pw_coverage_full[tuple(seqs.keys())] = aligned_proportion

    to_trim = []  # Indices of columns to trim
    under_co_cnt = []

    # Multi-thread column assessment for trimming
    col_assess_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_threads)
    to_trim_lock = threading.Lock()  # For appending cols to trim
    under_co_lock = (
        threading.Lock()
    )  # For keeping track of # of cols assessed for trimming

    # Method for assessing columns for potential trimming
    def assess_col(i):

        # Check if column occupancy is below base_threshold and assessable for trimming
        if 1 - (full_aln[:, i].count("-") / len(full_aln)) < base_threshold:
            with under_co_lock:
                under_co_cnt.append(i)

            # Collect seqs with content in column i
            sub_aln = [
                full_aln[j] for j in range(len(full_aln)) if full_aln[j][i] != "-"
            ]

            # Sort coverage comparisons by whether both seqs have content in column of interest
            pw_coverage_pos = {}  # Both seqs have content in column of interest
            pw_coverage_neg = {}  # One or both do not
            sub_pairs = (
                []
            )  # List of immutable sets representing pairs of seqs in sub_aln
            for j in range(len(sub_aln) - 1):
                for k in range(j, len(sub_aln)):
                    sub_pairs.append(tuple([sub_aln[j].name, sub_aln[k].name]))
            for seqs, coverage in pw_coverage_full.items():
                if seqs in sub_pairs:
                    pw_coverage_pos[seqs] = coverage
                else:
                    pw_coverage_neg[seqs] = coverage

            # Maintain column only if positive pairwise coverages are significantly higher than negatives
            t, p = stats.ttest_ind(
                list(pw_coverage_pos.values()),
                list(pw_coverage_neg.values()),
                equal_var=False,
                alternative="greater",
            )
            if (
                p >= p_threshold
            ):  # Trim if NOT significantly higher coverage in aligned seqs
                with to_trim_lock:
                    to_trim.append(i)

            print(f"Assessed col {i}")

        else:
            print(f"Col {i} above CO threshold")

    future_store = [
        col_assess_executor.submit(assess_col, i)
        for i in range(full_aln.get_alignment_length())
    ]
    concurrent.futures.wait(future_store)

    # TODO: Now in nested function for multi-threading - remove if all works
    # for i in range(full_aln.get_alignment_length()):   # For each column

    # # Check if column occupancy is below base_threshold and assessable for trimming
    # if 1 - (full_aln[:, i].count('-') / len(full_aln)) < base_threshold:
    #     under_co_cnt += 1
    #
    #
    #     # Collect seqs with content in column i
    #     sub_aln = [full_aln[j] for j in range(len(full_aln)) if full_aln[j][i] != '-']
    #
    #     # Sort coverage comparisons by whether both seqs have content in column of interest
    #     pw_coverage_pos = {}  # Both seqs have content in column of interest
    #     pw_coverage_neg = {}  # One or both do not
    #     sub_pairs = []  # List of immutable sets representing pairs of seqs in sub_aln
    #     for j in range(len(sub_aln)-1):
    #         for k in range(j, len(sub_aln)):
    #             # TODO: Check if below line is correct (was throwing error, have made a change but not sure its right)
    #             sub_pairs.append(tuple([sub_aln[j].name, sub_aln[k].name]))
    #     for seqs, coverage in pw_coverage_full.items():
    #         if seqs in sub_pairs:
    #             pw_coverage_pos[seqs] = coverage
    #         else:
    #             pw_coverage_neg[seqs] = coverage
    #
    #     # Maintain column only if positive pairwise coverages are significantly higher than negatives
    #     t, p = stats.ttest_ind(list(pw_coverage_pos.values()), list(pw_coverage_neg.values()), equal_var=False,
    #                            alternative='greater')
    #     if p >= p_threshold:  # Trim if NOT significantly higher coverage in aligned seqs
    #         to_trim.append(i)
    #
    #         trimmed_cnt += 1
    #
    #     print(f'Assessed col {i}')
    #
    # else:
    #     print(f'Col {i} above CO threshold')

    # Trim appropriate columns
    trimmed = trim_aln(full_aln, to_trim)

    print(f"Trimmed {len(to_trim)} of {len(under_co_cnt)} under occupancy threshold.")

    # Trimmed cols may be out of order due to multi-threading
    to_trim.sort()

    print(to_trim)

    # Write trimmed aln to file if specified, else return MSA object
    if out_file:
        AlignIO.write(trimmed, out_file, format)
        #         return to_trim   # Return trimmed cols as list
        return to_trim
    else:
        return trimmed


# -

infile = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/profile_creation/alns/gb1_30%_sim_nr90_high_acc.aln"
trimmed = smart_trim(
    in_file=infile, base_threshold=0.3
)  # , out_file="gb1_30%_sim_nr90_high_acc_trimmed_0.05.aln")

orig_aln = st.read_aln_file(
    "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/profile_creation/alns/gb1_30%_sim_nr90_high_acc.aln"
)
orig_aln.head()

wt = orig_aln["sequence"][0]  # grab the wild type as well
# columns that have been removed by smart_trim from the original alignment.
trimmed = [
    0,
    3,
    39,
    45,
    46,
    47,
    52,
    57,
    64,
    71,
    72,
    86,
    87,
    88,
    89,
    90,
    98,
    100,
    101,
    102,
    104,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    153,
    155,
    160,
    161,
    163,
    164,
    174,
    179,
    180,
    181,
    186,
    187,
    198,
    199,
    200,
    242,
    243,
    276,
    297,
    298,
    307,
    308,
    328,
    329,
    339,
    357,
    358,
    365,
    366,
    372,
    373,
    374,
    376,
    384,
    385,
    386,
    389,
    396,
    399,
    404,
    406,
    414,
    426,
    427,
    428,
    430,
    433,
    434,
    435,
    436,
    447,
    449,
    469,
    525,
    526,
    528,
    533,
    552,
    581,
    582,
    583,
    596,
    602,
    603,
    630,
    631,
    660,
    669,
    701,
    712,
    713,
    723,
    739,
    754,
    755,
    756,
    798,
    799,
    806,
    810,
    816,
    817,
    818,
    821,
    823,
    831,
    832,
    833,
    892,
    915,
    916,
    917,
    924,
    932,
    933,
    934,
    965,
    1022,
    1024,
    1025,
    1026,
    1027,
    1028,
    1031,
    1032,
    1033,
    1034,
    1088,
    1109,
    1110,
    1111,
    1130,
    1131,
    1214,
    1215,
    1216,
    1222,
    1230,
    1265,
    1266,
    1273,
    1295,
    1296,
    1297,
    1347,
    1348,
    1354,
    1365,
    1374,
    1393,
    1402,
    1404,
    1410,
    1417,
    1418,
    1424,
    1447,
    1463,
    1473,
    1486,
]

# read in the variants
vars = pd.read_csv(
    "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/SPG1_STRSG_Wu_2016.csv"
)
vars.head()

# make into a numpy msa
vars.rename(columns={"mutant": "id", "mutated_sequence": "sequence"}, inplace=True)
vars_msa, _, _ = st.convert_msa_numpy_array(vars)

# +
# we need to resize our variants to match the WT in the original alignment
resized_variants = np.zeros((vars_msa.shape[0], len(wt)))

# go through the wt, every time we see content, we know we can put the variant information
# at this column as we're only making substituion mutations.
var_position = 0
for idx, residue in enumerate(wt):
    if residue != "-":
        resized_variants[:, idx] = vars_msa[:, var_position]
        var_position += 1

resized_variants.shape

# +
from MAP_VAE.utils.seq_tools import IDX_TO_AA

# save the aligned variants before trimming
seqs = []
ids = []
for i in range(resized_variants.shape[0]):
    seq = "".join([IDX_TO_AA[x] for x in resized_variants[i, :]])
    seqs.append(seq)
    ids.append(vars["id"][i])

var_fasta = pd.DataFrame({"id": ids, "sequence": seqs})
st.write_fasta_file("gb1_variants.aln", var_fasta)

# +

# now trim the variants to match the alignment produced by smart_trim
non_trimmed_indices = [x for x in range(resized_variants.shape[1]) if x not in trimmed]
trimmed_variants = resized_variants[:, non_trimmed_indices]

seqs = []
ids = []
for i in range(trimmed_variants.shape[0]):
    seq = "".join([IDX_TO_AA[x] for x in trimmed_variants[i, :]])
    seqs.append(seq)
    ids.append(vars["id"][i])

var_fasta = pd.DataFrame({"id": ids, "sequence": seqs})
st.write_fasta_file("gb1_variants_trimmed.aln", var_fasta)
