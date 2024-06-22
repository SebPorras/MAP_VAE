# evoVAE


## Installation

Activate your conda environment and then run. 

Install with `pip install -e .`

## Training the model 

To run the model, you must provide a config file.

Example config files are found in `./scripts/configs/`

Currently, there are example configs for GB1, GCN4, A4 & GFP

```
AA_count: 21
alignment: /scratch/user/s4646506/gb1/encoded_weighted/gb1_ancestors_extants_no_dupes.pkl
architecture: SeqVAEv2
batch_size: 256
dms_file: /scratch/user/s4646506/gb1/dms_data/SPG1_STRSG_Wu_2016.pkl
dms_id: SPG1_STRSG_Wu_2016
dms_metadata: /scratch/user/s4646506/evoVAE/data/DMS_substitutions.csv
dropout: 0.025
epochs: 500
extant_aln: /scratch/user/s4646506/gb1/encoded_weighted/gb1_extants_no_dupes.pkl
hidden_dims:
- 150
- 150
info: ./gb1_ae
latent_dims: 3
learning_rate: 0.01
max_mutation: 4
max_norm: -1
num_processes: 8
patience: 3
project: null
replicate_csv: null
seq_theta: 0.2
test_split: 0.2
weight_decay: 0
```

##### Important config info. 
alignment: specifies where the pickled training sequences are kept. 
dms_id: This must match the id found in `data/DMS_substitutions.csv`
info: this specifies the name of the output directory that will be created. 

To start training, run the following command. 

`python train_vae.py gb1_ancestors_extants_config.yaml`

All ouptut will be written to a folder called `./gb1_ae/` as specified by the info line in your config file. 

If you want to specifiy a particular replicate, use this command, where 3 is the number of your particular replicate: 

`python train_vae.py gb1_ancestors_extants_config.yaml 3`

 Now the output folder will look like `./gb1_ae_r3/`.

`train_vae.py` currently uses the SeqVAE model found in `evoVAE/models/seqVAE.py`



