import evoVAE.utils.seq_tools as st
import random
import pandas as pd

seq_count = 10000
seq_len = 238
seqs = []
ids = []
count = 0
for i in range(1, seq_count+1):
    rand_seq = "".join([st.IDX_TO_AA[random.randint(0, 20)] for x in range(seq_len)])
    seqs.append(rand_seq)
    ids.append(f"{i}_rand")
 
df = pd.DataFrame({"id": ids, "sequence": seqs})
st.write_fasta_file("gfp_random_seqs.aln", df)

