import yaml

groups = ["small_ancestors_extants", "small_ancestors"]
ids = ["ae", "a"]

for g, id in zip(groups, ids):

    data = {
        "alignment": f"/scratch/user/s4646506/gb1_small/alns/gb1_{g}_no_dupes.aln",
        "extant_aln": "/scratch/user/s4646506/gb1_small/alns/gb1_extants_no_dupes.aln",
        "AA_count": 21,
        "info": "./gb1_" + id,
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
        "dms_file": "/scratch/user/s4646506/gb1/dms_data/SPG1_STRSG_Wu_2016.csv",
        "dms_metadata": "/scratch/user/s4646506/evoVAE/data/DMS_substitutions.csv",
        "dms_id": "SPG1_STRSG_Wu_2016",
        "replicate_csv": None,
    }

    yaml_str = yaml.dump(data, default_flow_style=False)

    with open(f"gb1_{g}_config.yaml", "w") as file:
        file.write(yaml_str)
