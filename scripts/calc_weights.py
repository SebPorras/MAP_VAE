import evoVAE.utils.seq_tools as st

all_aln = st.read_aln_file("gfp_ancestors_extants.aln")

encodings, weights = st.encode_and_weight_seqs(all_aln["sequence"], theta=0.2)

all_aln["weights"] = weights
all_aln["encodings"] = encodings

all_aln.to_pickle("gfp_ancestors_extants.pkl")
