import yaml

lrs = [0.01, 0.001, 0.0001]
batch = [32, 64, 128, 256]
data = [
    "gb1_ancestors_encoded_weighted_no_dupes.pkl",
]
dropout = [0.05, 0.025, 0.01]
replicates = 1

ARRAY_ID = 2
ALIGN_FILE = -2

count = 0
for lr in lrs:
    for b in batch:
        for d in dropout:
            for r in range(1, replicates + 1):

                data = {
                    "alignment": "/scratch/user/s4646506/gb1/encoded_weighted/gb1_ancestors_encoded_weighted_no_dupes.pkl",
                    "extant_aln": "/scratch/user/s4646506/gb1/encoded_weighted/gb1_extants_encoded_weighted_no_dupes.pkl",
                    "info": "None",
                    "project": "GB1_tuning_round_2",
                    "seq_theta": 0.2,  # reweighting,
                    "test_split": 0.2,
                    "max_mutation": 4,  # how many mutations the model will test up to,
                    "learning_rate": lr,  # ADAM,
                    "weight_decay": 0,  # ADAM,
                    "dropout": d,
                    "epochs": 150,
                    "batch_size": b,
                    "patience": 20,
                    "max_norm": -1,  # gradient clipping,
                    "architecture": "SeqVAE_softmax",
                    "latent_dims": 3,
                    "hidden_dims": [256, 128, 64],
                    "dms_file": "/scratch/user/s4646506/gb1/dms_data/SPG1_STRSG_Wu_2016.pkl",
                    "dms_metadata": "/scratch/user/s4646506/evoVAE/data/DMS_substitutions.csv",
                    "dms_id": "SPG1_STRSG_Wu_2016",
                    "replicate": r,
                    "latent_samples": 50,
                    "num_processes": 8,
                }

                # example file name would look like  eg. /data/gb1_ancestors.aln
                training_aln = data["alignment"].split("/")[-1].split(".")[ALIGN_FILE]
                # alignment to calculate covariances on. eg. /data/gb1_extants.pkl
                covar_aln = data["extant_aln"].split("/")[-1].split(".")[ALIGN_FILE]

                unique_id = f'./lr_{data["learning_rate"]}_b_{data["batch_size"]}_dropout_{data["dropout"]}_rep_{data["replicate"]}_train_{training_aln}_dms_{data["dms_id"]}_covar_{covar_aln}/'
                data["info"] = unique_id

                yaml_str = yaml.dump(data, default_flow_style=False)

                with open(f"{count}.yaml", "w") as file:
                    file.write(yaml_str)

                count += 1
