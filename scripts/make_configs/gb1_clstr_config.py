import yaml


props = [0.0, 0.05, 0.1, 0.15]

for p in props:

    data = {
        "alignment": f"/scratch/user/s4646506/gb1/encoded_weighted/gb1_ancestors_extants_no_dupes.pkl",
        "extant_aln": "/scratch/user/s4646506/gb1/encoded_weighted/gb1_extants_no_dupes.pkl",
        "AA_count": 21,
        "info": f"./gb1_{p}",
        "project": None,
        "seq_theta": 0.2,  # reweighting,
        "test_split": 0.2,
        "max_mutation": 4,  # how many mutations the model will test up to,
        "learning_rate": 0.0001,  # ADAM,
        "weight_decay": 0,  # ADAM,
        "epochs": 300,
        "batch_size": 128,
        "patience": 3,
        "max_norm": -1,  # gradient clipping,
        "architecture": "SeqVAEv2",
        "latent_dims": 3,
        "hidden_dims": [150, 150],
        "dms_file": "/scratch/user/s4646506/gb1/dms_data/SPG1_STRSG_Wu_2016.csv",
        "dms_metadata": "/scratch/user/s4646506/evoVAE/data/DMS_substitutions.csv",
        "dms_id": "SPG1_STRSG_Wu_2016",
        "replicate_csv": f"/scratch/user/s4646506/gb1/gb1_{p}_replicates.csv",
    }

    yaml_str = yaml.dump(data, default_flow_style=False)

    with open(f"gb1_{p}_config.yaml", "w") as file:
        file.write(yaml_str)
