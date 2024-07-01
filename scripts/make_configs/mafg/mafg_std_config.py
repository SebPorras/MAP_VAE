import yaml

groups = ["ancestors_extants", "ancestors", "extants"]
ids = ["ae", "a", "e"]

for g, id in zip(groups, ids):

    data = {
        "alignment": f"/scratch/user/s4646506/mafg/encoded_weighted/mafg_{g}_no_dupes.fasta",
        "extant_aln": "/scratch/user/s4646506/mafg/encoded_weighted/mafg_extants_no_dupes.fasta",
        "AA_count": 21,
        "info": "./mafg_" + id,
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
        "dms_file": "/scratch/user/s4646506/mafg/dms_data/MAFG_MOUSE_Tsuboyama_2023_1K1V.csv",
        "dms_metadata": "/scratch/user/s4646506/evoVAE/data/DMS_substitutions.csv",
        "dms_id": "MAFG_MOUSE_Tsuboyama_2023_1K1V",
        "replicate_csv": None,
    }

    yaml_str = yaml.dump(data, default_flow_style=False)

    with open(f"mafg_{g}_config.yaml", "w") as file:
        file.write(yaml_str)
