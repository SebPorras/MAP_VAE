import yaml


props = [0.0, 0.1, 0.2, 0.05, 0.15]

for p in props:

    data = {
        "alignment": f"/scratch/user/s4646506/a4/encoded_weighted/a4_ancestors_extants_no_dupes.pkl",
        "extant_aln": "/scratch/user/s4646506/a4/encoded_weighted/a4_extants_no_dupes.pkl",
        "AA_count": 21,
        "info": f"./a4_{p}_extants",
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
        "dms_file": "/scratch/user/s4646506/a4/dms_data/A4_HUMAN_Seuma_2022.pkl",
        "dms_metadata": "/scratch/user/s4646506/evoVAE/data/DMS_substitutions.csv",
        "dms_id": "A4_HUMAN_Seuma_2022",
        "replicate_csv": f"/scratch/user/s4646506/a4/a4_{p}_replicates.csv",
        "num_processes": 8,
    }

    yaml_str = yaml.dump(data, default_flow_style=False)

    with open(f"a4_{p}_config.yaml", "w") as file:
        file.write(yaml_str)
