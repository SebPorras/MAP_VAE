import evoVAE.utils.seq_tools as st
import pandas as pd

#all_aln = st.read_aln_file("../data/gb1/gb1_ancestors.aln")

all_aln = pd.read_pickle("../data/gfp/GFP_AEQVI_full_04-29-2022_b08_ancestors_no_syn.pkl")
print(all_aln.shape)
all_aln = all_aln.drop_duplicates(subset=['sequence'])
print(all_aln.shape)

encodings, weights = st.encode_and_weight_seqs(all_aln["sequence"], theta=0.2)

all_aln["weights"] = weights

all_aln.to_pickle("GFP_AEQVI_full_04-29-2022_b08_ancestors_no_syn_no_dupes.pkl")


##########
all_aln = pd.read_pickle("../data/gfp/GFP_AEQVI_full_04-29-2022_b08_ancestors_extants_no_syn.pkl")

print(all_aln.shape)
all_aln = all_aln.drop_duplicates(subset=['sequence'])

print(all_aln.shape)
encodings, weights = st.encode_and_weight_seqs(all_aln["sequence"], theta=0.2)

all_aln["weights"] = weights

all_aln.to_pickle("GFP_AEQVI_full_04-29-2022_b08_ancestors_extants_no_syn_no_dupes.pkl")

##########


all_aln = pd.read_pickle("../data/gfp/GFP_AEQVI_full_04-29-2022_b08_extants_no_syn.pkl")
print(all_aln.shape)
all_aln = all_aln.drop_duplicates(subset=['sequence'])
print(all_aln.shape)

encodings, weights = st.encode_and_weight_seqs(all_aln["sequence"], theta=0.2)

all_aln["weights"] = weights

all_aln.to_pickle("GFP_AEQVI_full_04-29-2022_b08_extants_no_syn_no_dupes.pkl")





"""
df = pd.read_csv("/scratch/user/s4646506/gb1/SPG1_STRSG_Wu_2016.csv")
encodings, weight = st.encode_and_weight_seqs(df['mutated_sequence'], 0.2, reweight=False)
df['encoding'] = encodings
df.to_pickle("SPG1_STRSG_Wu_2016.pkl")

encodings, weights = st.encode_and_weight_seqs(all_aln["sequence"], theta=0.2)

all_aln["weights"] = weights
#all_aln["encoding"] = encodings

all_aln.to_pickle("gb1_ancestors_encoded_weighted.pkl")
"""
