import yaml


props = [0.0, 0.02, 0.07, 0.12, 0.17, 0.22]

for p in props:

    data = {
        "alignment": f"/scratch/user/s4646506/gb1/encoded_weighted/gb1_ancestors_extants_no_dupes.pkl",
        "extant_aln": "/scratch/user/s4646506/gb1/encoded_weighted/gb1_extants_no_dupes.pkl",
        "AA_count": 21,
        "info": f"./gb1_{p}_extants_sanjana",
        "project": None,
        "seq_theta": 0.2,  # reweighting,
        "test_split": 0.2,
        "max_mutation": 4,  # how many mutations the model will test up to,
        "learning_rate": 0.01,  # ADAM,
        "weight_decay": 0,  # ADAM,
        "dropout": 0.025,
        "epochs": 200,
        "batch_size": 128,
        "patience": 3,
        "max_norm": -1,  # gradient clipping,
        "architecture": "SeqVAEv2",
        "latent_dims": 3,
        "hidden_dims": [150, 150],
        "dms_file": "/scratch/user/s4646506/gb1/dms_data/SPG1_STRSG_Wu_2016.pkl",
        "dms_metadata": "/scratch/user/s4646506/evoVAE/data/DMS_substitutions.csv",
        "dms_id": "SPG1_STRSG_Wu_2016",
        "replicate_csv": f"/scratch/user/s4646506/gb1/sanjana_reps/gb1_{p}_replicates_sanjana.csv",
        "num_processes": 8,
    }

    yaml_str = yaml.dump(data, default_flow_style=False)

    with open(f"gb1_{p}_config_sanjana.yaml", "w") as file:
        file.write(yaml_str)
