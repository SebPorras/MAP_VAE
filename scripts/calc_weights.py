import evoVAE.utils.seq_tools as st

all_aln = st.read_aln_file("/Users/sebs_mac/OneDrive - The University of Queensland/honours/data/gfp_alns/independent_runs/no_synthetic/alns/GFP_AEQVI_full_04-29-2022_b08_extants_no_syn.aln")

encodings, weights = st.encode_and_weight_seqs(all_aln["sequence"], theta=0.2)

all_aln["weights"] = weights
all_aln["encodings"] = encodings

all_aln.to_pickle("test.pkl")
