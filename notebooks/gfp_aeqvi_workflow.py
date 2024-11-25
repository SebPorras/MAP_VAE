# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### GFP_AEQV processing

# +
import MAP_VAE.utils.seq_tools as st
from MAP_VAE.utils.webservice import get_sequence_batch
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
from ete3 import Tree
from sequence import *

DATA_DIR = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/"

aln = st.read_aln_file(DATA_DIR + "SPG1_STRSG_full_b0.1.fasta", encode=False)
# -

#

# # Reading in alignment

# +

# st.write_fasta_file(f"../data/alignments/gfp_alns/GFP_AEQVI_full_04-29-2022_b08.aln", aln)

# draws = []
# for draw in range(100):
#    draws.append(aln.sample(n=250, random_state=draw, replace=False))

# for i, seqs in enumerate(draws):
#    st.write_fasta_file(f"../data/alignments/gfp_alns/GFP_AEQVI_full_04-29-2022_b08_rand_subset_{i}.aln", seqs)
# aln
# -

# # Collecting annotations about sequences

# +
ids = aln["id"]
ids = [x for x in ids.apply(lambda x: x.split("/")[0]).values]

print([x.split("_")[1] for x in ids])
# Sequences found in the very distant branch of the tree.
outgroup = [
    "UniRef100_Q6RYS7/1-232",
    "UniRef100_UPI000B5ABD49/2-234",
    "UniRef100_A0A5J6CYI5/1-229",
    "UniRef100_Q6RYS6/40-259",
    "UniRef100_Q6RYS5/2-223",
    "UniRef100_J9PIH6/6-230",
    "UniRef100_J9PJD5/7-231",
    "UniRef100_A0A7M5TYI8/7-231",
    "UniRef100_A0A5J6CYV7/1-229",
    "UniRef100_A0A5J6CYI7/1-229",
    "UniRef100_A0A5J6CYT5/1-229",
    "UniRef100_A0A5J6CYV4/1-229",
    "UniRef100_A0A5J6CYK8/1-229",
    "UniRef100_A0A7M5TUN8/3-229",
    "UniRef100_A0A7M5UKF9/3-229",
    "UniRef100_A0A7M5WQI9/3-229",
    "UniRef100_A0A7M5UPI8/3-229",
    "UniRef100_A0A7M5V224/3-229",
    "UniRef100_J9PGT0/3-229",
    "UniRef100_A0A7M5WWU1/3-229",
    "UniRef100_A0A7M5XGN4/3-229",
    "UniRef100_G1JSF2/3-231",
    "UniRef100_G1JSF3/3-231",
    "UniRef100_G1JSF4/3-231",
    "UniRef100_A0A7M5WWD0/7-234",
    "UniRef100_A0A7M5V407/7-234",
    "UniRef100_A0A7M6DMT2/54-281",
    "UniRef100_J9PGG2/7-234",
    "UniRef100_A0A7M5X361/50-277",
    "UniRef100_D7PM05/5-232",
    "UniRef100_D7PM10/5-232",
    "UniRef100_D7PM12/5-232",
    "UniRef100_D7PM04/5-232",
    "UniRef100_D7PM06/5-232",
]

ID = 0
SPECIES = 1
# in-group sequences
outgroup = [x.split("/")[ID] for x in outgroup]
print(len(outgroup))
# FIX ME, querying for uniref seqs is still broken, skipping first uniprot entry
ingroup = [x for x in ids[1:] if x not in outgroup]
print(len(ingroup))

uniprot_cols = ["id", "organism"]
outgroup_query = get_sequence_batch(
    outgroup, uniprot_cols, query_field="id", database="uniref"
)
ingroup_query = get_sequence_batch(
    ingroup, uniprot_cols, query_field="id", database="uniref"
)

outgroup_species = pd.Series([x[SPECIES] for x in outgroup_query]).value_counts()
ingroup_species = pd.Series([x[SPECIES] for x in ingroup_query]).value_counts()
# -

# # Removing synthetic sequences

# +
synthetic_seqs = [x for x, y in ingroup_query if y == "synthetic construct"]

no_syn_aln_seqs = []
all_seqs = []
for name, seq in zip(aln["id"], aln["sequence"]):

    if name.split("/")[0] not in synthetic_seqs:
        x = Sequence(seq, name=name, gappy=True)
        no_syn_aln_seqs.append(x)

    x = Sequence(seq, name=name, gappy=True)
    all_seqs.append(x)

print(len(no_syn_aln_seqs), len(all_seqs))
writeFastaFile(
    filename="no_synthetic_GFP_AEQVI_full_04-29-2022_b08.aln", seqs=no_syn_aln_seqs
)

# +
all_aln = Alignment(all_seqs)
no_syn_aln = Alignment(no_syn_aln_seqs)

con_all = all_aln.getConsensus()
no_syn_con = no_syn_aln.getConsensus()
count = 0
for x, y in zip(no_syn_con, con_all):

    if x != y:
        print(f"Difference at : {count}")

    count += 1

# -

outgroup_species

ingroup_species

# # Rooting tree

# Minimum ancestor deviation (MAD) will be used to root the tree as all sequences are assumed to be homologs.

# # Placing ancestors and extants in one file

# +
ANC_DIR = "/Users/sebs_mac/OneDrive - The University of Queensland/honours/data/gfp_alns/independent_runs/no_synthetic/ancestors/auto_rooted/"
DATA_DIR = "/Users/sebs_mac/OneDrive - The University of Queensland/honours/data/gfp_alns/independent_runs/no_synthetic/alns/"

# trees rejected by AU test or failed run
bad_trees = [11, 31, 48, 60, 78, 63]
# don't use ancestors from the outgroup.
# bad_ancestors = ['N' + str(i) for i in range(0,34)]
anc_extants = []
anc_only = []


# EXTANTS
aln = st.read_aln_file(DATA_DIR + "GFP_AEQVI_full_04-29-2022_b08_extants_no_syn.aln")
for name, seq in zip(aln["id"], aln["sequence"]):
    x = Sequence(seq, name=name, gappy=True)
    anc_extants.append(x)

# ANCESTORS
for i in range(1, 101):

    if i in bad_trees:
        continue

    aln = st.read_aln_file(ANC_DIR + f"/ancestors/run_{i}_ancestors.fa", encode=False)
    for name, seq in zip(aln["id"], aln["sequence"]):

        seq_name = name + f"_tree_{i}"
        x = Sequence(seq, name=seq_name, gappy=True)
        anc_extants.append(x)
        anc_only.append(x)

OUT_DIR = "/Users/sebs_mac/OneDrive - The University of Queensland/honours/data/gfp_alns/independent_runs/no_synthetic/alns/"
writeFastaFile(
    OUT_DIR + "GFP_AEQVI_full_04-29-2022_b08_ancestors_extants_no_syn.aln", anc_extants
)
writeFastaFile(OUT_DIR + "GFP_AEQVI_full_04-29-2022_b08_ancestors_no_syn.aln", anc_only)
