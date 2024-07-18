# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: embed
#     language: python
#     name: python3
# ---

# %%
import evoVAE.utils.seq_tools as st
import numpy as np
from sequence import *
from sym import *
import pandas as pd
from evoVAE.utils.seq_tools import write_fasta_file


# %%
def calc_pairwise_similarity(seq_path, wt_seq, wt_name):
    
    seqs = readFastaFile(seq_path)
    b62 = readSubstMatrix('../data/blosum62.matrix', Protein_Alphabet)

    wt = Sequence(sequence=wt_seq, 
                name=wt_name)

    results = []
    for seq in seqs:

        aln = alignGlobal(seq, wt, b62)

        wt_aln = aln.seqs[1]
        query = aln.seqs[0]

        count = 0 
        for x, y in zip(wt_aln, query):
            if x == y and x != "-" and y != "-":
                count += 1

        results.append(count / len(wt_aln))

    return results


# %% [markdown]
# # GB1 - removing low identity sequences

# %%
seqs = readFastaFile("/Users/sebs_mac/uni_OneDrive/honours/data/gb1/alns/gb1_all_extants.fasta")
b62 = readSubstMatrix('../data/blosum62.matrix', Protein_Alphabet)

wt = Sequence(sequence="MEKEKKVKYFLRKSAFGLASVSAAFLVGSTVFAVDSPIEDTPIIRNGGELTNLLGNSETTLALRNEESATADLTAAAVADTVAAAAAENAGAAAWEAAAAADALAKAKADALKEFNKYGVSDYYKNLINNAKTVEGIKDLQAQVVESAKKARISEATDGLSDFLKSQTPAEDTVKSIELAEAKVLANRELDKYGVSDYHKNLINNAKTVEGVKELIDEILAALPKTDQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTEKPEVIDASELTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEMVTEVPGDAPTEPEKPEASIPLVPLTPATPIAKDDAKKDDTKKEDAKKPEAKKDDAKKAETLPTTGEGSNPFFTAAALAVMAGAGALAVASKRKED", 
              name="SPG1_STRSG")

results = []
for seq in seqs:

    aln = alignGlobal(seq, wt, b62)

    wt_aln = aln.seqs[1]
    query = aln.seqs[0]

    count = 0 
    for x, y in zip(wt_aln, query):
        if x == y and x != "-" and y != "-":
            count += 1

    results.append(count / len(wt_aln))

# %%

# %%
orignal_aln = st.read_aln_file("/Users/sebs_mac/uni_OneDrive/honours/data/gb1/alns/gb1_all_extants.aln")

orignal_aln["similarity"] = results
trimmed = orignal_aln[orignal_aln["similarity"] >= 0.32]
trimmed.drop_duplicates(subset=["sequence"], inplace=True)
trimmed

# %% [markdown]
# # GCN4 - removing low identify sequences

# %%
orignal_fasta = st.read_aln_file("/Users/sebs_mac/uni_OneDrive/honours/data/gcn4/alns/gcn4_extants.fasta")
wt = orignal_fasta[orignal_fasta["id"] == "GCN4_YEAST/1-281"]

sim = calc_pairwise_similarity("/Users/sebs_mac/uni_OneDrive/honours/data/gcn4/alns/gcn4_extants.fasta",
                         wt["sequence"][0], "GCN4_YEAST/1-281")

# %%
orignal_fasta["similarity"] = sim
trimmed = orignal_fasta[orignal_fasta["similarity"] >= 0.3]
#trimmed.drop_duplicates(subset=["sequence"], inplace=True)

# %%
aln = st.read_aln_file("/Users/sebs_mac/uni_OneDrive/honours/data/gcn4/alns/gcn4_extants.aln")
aln = aln[aln["id"].isin(trimmed["id"])]


# %%
st.write_fasta_file("gcn4_0.3_sim_extants_no_dupes.aln", aln)

# %% [markdown]
# # A4

# %%
orignal_fasta = st.read_aln_file("/Users/sebs_mac/uni_OneDrive/honours/data/a4_human/alns/a4_extants.fasta")

wt = orignal_fasta[orignal_fasta["id"] == "A4_HUMAN/1-770"]
sim = calc_pairwise_similarity("/Users/sebs_mac/uni_OneDrive/honours/data/a4_human/alns/a4_extants.fasta",
                          wt["sequence"][0], "A4_HUMAN/1-770")

# %%
orignal_fasta["similarity"] = sim


# %%
trimmed = orignal_fasta[orignal_fasta["similarity"] >= 0.8]
trimmed.drop_duplicates(subset=["sequence"], inplace=True)

original_aln = st.read_aln_file("/Users/sebs_mac/uni_OneDrive/honours/data/a4_human/alns/a4_extants.aln")
original_aln = original_aln[original_aln["id"].isin(trimmed["id"])]

st.write_fasta_file("a4_0.8_sim_extants.aln",original_aln)

# %%
original_aln

# %%

# %%
