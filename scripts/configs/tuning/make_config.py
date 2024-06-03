import yaml

"""
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
                    "AA_count": 21,
                    "info": "None",
                    "project": "GB1_tuning_round_2",
                    "seq_theta": 0.2,  # reweighting,
                    "test_split": 0.2,
                    "max_mutation": 4,  # how many mutations the model will test up to,
                    "learning_rate": lr,  # ADAM,
                    "weight_decay": 0,  # ADAM,
                    "dropout": d,
                    "epochs": 800,
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
"""


# GB1
replicates = 5

ARRAY_ID = 2
ALIGN_FILE = -2
extant_per = [0, 0.02, 0.07, 0.12, 0.17, 0.22]
count = 0

for e in extant_per:
    for r in range(1, replicates + 1):

        data = {
            "alignment": f"/scratch/user/s4646506/gb1/encoded_weighted/gb1_ancestors_extants_no_dupes_clustered_r{r}_extant_{e}_encoded_weighted.pkl",
            "extant_aln": "/scratch/user/s4646506/gb1/encoded_weighted/gb1_extants_encoded_weighted.pkl",
            "AA_count": 21,
            "info": "None",
            "project": "gb1_clustering",
            "seq_theta": 0.2,  # reweighting,
            "test_split": 0.2,
            "max_mutation": 4,  # how many mutations the model will test up to,
            "learning_rate": 0.01,  # ADAM,
            "weight_decay": 0,  # ADAM,
            "dropout": 0.01,
            "epochs": 800,
            "batch_size": 256,
            "patience": 10,
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

        unique_id = f'./rep_{data["replicate"]}_train_{training_aln}_dms_{data["dms_id"]}_covar_{covar_aln}/'
        data["info"] = unique_id

        yaml_str = yaml.dump(data, default_flow_style=False)

        with open(f"gb1_r{r}_extant_{e}.yaml", "w") as file:
            file.write(yaml_str)


# A4
# replicates = 5

# ARRAY_ID = 2
# ALIGN_FILE = -2
# extant_per = [0.0, 0.05, 0.1, 0.15, 0.2185]
# count = 0

# for e in extant_per:
#     for r in range(1, replicates + 1):

#         data = {
#             "alignment": f"/scratch/user/s4646506/a4/encoded_weighted/a4_ancestors_extants_no_dupes_clustered_r{r}_extant_{e}_encoded_weighted.pkl",
#             "extant_aln": "/scratch/user/s4646506/a4/encoded_weighted/a4_extants_encoded_weighted.pkl",
#             "AA_count": 21,
#             "info": "None",
#             "project": "A4_human",
#             "seq_theta": 0.2,  # reweighting,
#             "test_split": 0.2,
#             "max_mutation": 4,  # how many mutations the model will test up to,
#             "learning_rate": 0.01,  # ADAM,
#             "weight_decay": 0,  # ADAM,
#             "dropout": 0.01,
#             "epochs": 800,
#             "batch_size": 256,
#             "patience": 10,
#             "max_norm": -1,  # gradient clipping,
#             "architecture": "SeqVAE_softmax",
#             "latent_dims": 3,
#             "hidden_dims": [256, 128, 64],
#             "dms_file": "/scratch/user/s4646506/a4/dms_data/A4_HUMAN_Seuma_2022.pkl",
#             "dms_metadata": "/scratch/user/s4646506/evoVAE/data/DMS_substitutions.csv",
#             "dms_id": "A4_HUMAN_Seuma_2022",
#             "replicate": r,
#             "latent_samples": 50,
#             "num_processes": 8,
#         }

#         # example file name would look like  eg. /data/gb1_ancestors.aln
#         training_aln = "".join(data["alignment"].split("/")[-1].split(".")[:-1])
#         # alignment to calculate covariances on. eg. /data/gb1_extants.pkl
#         covar_aln = data["extant_aln"].split("/")[-1].split(".")[ALIGN_FILE]

#         unique_id = f'./rep_{data["replicate"]}_train_{training_aln}_dms_{data["dms_id"]}_covar_{covar_aln}/'
#         data["info"] = unique_id

#         yaml_str = yaml.dump(data, default_flow_style=False)

#         with open(f"a4_human_r{r}_extant_{e}.yaml", "w") as file:
#             file.write(yaml_str)


# gcn4
# replicates = 5

# ARRAY_ID = 2
# ALIGN_FILE = -2
# extant_per = [0, 0.01, 0.025, 0.05, 0.0662]
# count = 0

# for e in extant_per:
#     for r in range(1, replicates + 1):

#         data = {
#             "alignment": f"/scratch/user/s4646506/gcn4/encoded_weighted/gcn4_ancestors_extants_no_dupes_clustered_r{r}_extant_{e}_encoded_weighted.pkl",
#             "extant_aln": "/scratch/user/s4646506/gcn4/encoded_weighted/gcn4_extants_encoded_weighted.pkl",
#             "AA_count": 21,
#             "info": "None",
#             "project": "gcn4_clustering",
#             "seq_theta": 0.2,  # reweighting,
#             "test_split": 0.2,
#             "max_mutation": 4,  # how many mutations the model will test up to,
#             "learning_rate": 0.01,  # ADAM,
#             "weight_decay": 0,  # ADAM,
#             "dropout": 0.01,
#             "epochs": 800,
#             "batch_size": 256,
#             "patience": 10,
#             "max_norm": -1,  # gradient clipping,
#             "architecture": "SeqVAE_softmax",
#             "latent_dims": 3,
#             "hidden_dims": [256, 128, 64],
#             "dms_file": "/scratch/user/s4646506/gcn4/dms_data/GCN4_YEAST_Staller_2018.pkl",
#             "dms_metadata": "/scratch/user/s4646506/evoVAE/data/DMS_substitutions.csv",
#             "dms_id": "GCN4_YEAST_Staller_2018",
#             "replicate": r,
#             "latent_samples": 50,
#             "num_processes": 8,
#         }

#         # example file name would look like  eg. /data/gb1_ancestors.aln
#         training_aln = "".join(data["alignment"].split("/")[-1].split(".")[:-1])
#         # alignment to calculate covariances on. eg. /data/gb1_extants.pkl
#         covar_aln = data["extant_aln"].split("/")[-1].split(".")[ALIGN_FILE]

#         unique_id = f'./rep_{data["replicate"]}_train_{training_aln}_dms_{data["dms_id"]}_covar_{covar_aln}/'
#         data["info"] = unique_id

#         yaml_str = yaml.dump(data, default_flow_style=False)

#         with open(f"gcn4_r{r}_extant_{e}.yaml", "w") as file:
#             file.write(yaml_str)
