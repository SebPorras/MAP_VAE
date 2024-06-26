import yaml


props = [0.0, 0.05, 0.1, 0.15, 0.2185]

for p in props:

    data = {
        "alignment": f"/scratch/user/s4646506/a4/encoded_weighted/a4_ancestors_extants_no_dupes.pkl",
        "extant_aln": "/scratch/user/s4646506/a4/encoded_weighted/a4_extants_no_dupes.pkl",
        "AA_count": 21,
        "info": f"./a4_{p}_sanjana_extants",
        "project": None,
        "seq_theta": 0.2,  # reweighting,
        "test_split": 0.2,
        "max_mutation": 4,  # how many mutations the model will test up to,
        "learning_rate": 0.0001,  # ADAM,
        "weight_decay": 0,  # ADAM,
        "epochs": 500,
        "batch_size": 128,
        "patience": 3,
        "architecture": "SeqVAEv2",
        "latent_dims": 3,
        "hidden_dims": [150, 150],
        "dms_file": "/scratch/user/s4646506/a4/dms_data/A4_HUMAN_Seuma_2022.pkl",
        "dms_metadata": "/scratch/user/s4646506/evoVAE/data/DMS_substitutions.csv",
        "dms_id": "A4_HUMAN_Seuma_2022",
        "replicate_csv": f"/scratch/user/s4646506/a4/sanjana_reps/a4_{p}_replicates_sanjana.csv",
    }

    yaml_str = yaml.dump(data, default_flow_style=False)

    with open(f"a4_{p}_config_sanjana.yaml", "w") as file:
        file.write(yaml_str)
