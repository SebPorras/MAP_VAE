import yaml

groups = ["ancestors_extants", "ancestors", "extants"]
ids = ["ae", "a", "e"]


for g, id in zip(groups, ids):

    data = {
        "alignment": f"/scratch/user/s4646506/gcn4/encoded_weighted/gcn4_{g}_no_dupes.pkl",
        "extant_aln": "/scratch/user/s4646506/gcn4/encoded_weighted/gcn4_extants_no_dupes.pkl",
        "AA_count": 21,
        "info": "./gcn4_" + id,
        "project": None,
        "seq_theta": 0.2,  # reweighting,
        "test_split": 0.2,
        "max_mutation": 4,  # how many mutations the model will test up to,
        "learning_rate": 0.0001,  # ADAM,
        "weight_decay": 0,  # ADAM,
        "epochs": 300,
        "batch_size": 128,
        "patience": 3,
        "architecture": "SeqVAEv2",
        "latent_dims": 3,
        "hidden_dims": [150, 150],
        "dms_file": "/scratch/user/s4646506/gcn4/dms_data/GCN4_YEAST_Staller_2018.csv",
        "dms_metadata": "/scratch/user/s4646506/evoVAE/data/DMS_substitutions.csv",
        "dms_id": "GCN4_YEAST_Staller_2018",
        "replicate_csv": None,
    }

    yaml_str = yaml.dump(data, default_flow_style=False)

    with open(f"gcn4_{g}_config.yaml", "w") as file:
        file.write(yaml_str)
