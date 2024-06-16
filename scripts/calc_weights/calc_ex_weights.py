import evoVAE.utils.seq_tools as st

all_aln = st.read_aln_file("../data/gb1/SPG1_STRSG_full_b0.1_filt.aln")
all_aln = all_aln.drop_duplicates(subset=['sequence'])

encodings, weights = st.encode_and_weight_seqs(all_aln["sequence"], theta=0.2)

all_aln["weights"] = weights
all_aln["encoding"] = encodings

all_aln.to_pickle("SPG1_STRSG_full_b0.1_filt_encoded_weighted_no_dupes.pkl")
