# %%
from evoVAE.utils.datasets import MSA_Dataset
import evoVAE.utils.seq_tools as st
from evoVAE.models.seqVAE import SeqVAE
from evoVAE.trainer.seq_trainer import seq_train
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import wandb


REPLICATES = 5
for i in range(1, REPLICATES + 1):
    # %% [markdown]
    # #### Config

    # %%
    wandb.init(
        project="GB1_ancestor_sampling",
        # hyperparameters
        config={"AA_count": 21,
                "alignment": f"/scratch/user/s4646506/gb1/random_sampling/data/gb1_ancestors_no_dupes_encoded_weighted_0.7_sample_r{i}.pkl",
                "architecture": "SeqVAE_reg_softmax",
                "batch_size": 64,
                "dms_file": "/scratch/user/s4646506/gb1/dms_data/SPG1_STRSG_Wu_2016.pkl",
                "dms_id": "SPG1_STRSG_Wu_2016",
                "dms_metadata": "/scratch/user/s4646506/evoVAE/data/DMS_substitutions.csv",
                "dropout": None,
                "epochs": 300,
                "hidden_dims": [256, 128,64],
                "info": "gb1_ancestors_no_dupes",
                "latent_dims": 3,
                "learning_rate": 0.001,
                "max_mutation": 4,
                "max_norm": 5,
                "momentum": None,
                "patience": 3,
                "project": "GB1_SeqVAE",
                "seq_theta": 0.2,
                "test_split": 0.2,
                "weight_decay": 0.0001},
    )


    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # %% [markdown]
    # #### Data loading and preprocessing

    # %%

    # Read in the datasets and create train and validation sets
    # Assume that encodings and weights have been calculated.
    # outgroup ancestors have already been removed.
    ancestors_aln = pd.read_pickle(config.alignment)
    train, val = train_test_split(ancestors_aln, test_size=config.test_split)

    # TRAINING
    train_dataset = MSA_Dataset(train["encoding"], train["weights"], train["id"])

    # VALIDATION
    val_dataset = MSA_Dataset(val["encoding"], val["weights"], val["id"])

    # DATA LOADERS #
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Load and process the DMS data used for fitness prediction
    dms_data = pd.read_pickle(config.dms_file)
    metadata = pd.read_csv(config.dms_metadata)
    # grab metadata for current experiment
    metadata = metadata[metadata["DMS_id"] == config.dms_id]

    # %% [markdown]
    # #### Create the model

    # %%
    # get the sequence length from first sequence
    SEQ_LEN = 0
    BATCH_ZERO = 0
    SEQ_ZERO = 0
    seq_len = train_dataset[BATCH_ZERO][SEQ_ZERO].shape[SEQ_LEN]
    input_dims = seq_len * config.AA_count

    # instantiate the model
    model = SeqVAE(
        input_dims=input_dims,
        latent_dims=config.latent_dims,
        hidden_dims=config.hidden_dims,
        config=config,
    )
    # model

    # %% [markdown]
    # #### Training Loop

    # %%
    trained_model = seq_train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        dms_data=dms_data,
        metadata=metadata,
        device=device,
        config=config,
    )

    # %%
    wandb.finish()
