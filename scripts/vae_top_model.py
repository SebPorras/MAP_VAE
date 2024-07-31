import torch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import scipy
import evoVAE.utils.seq_tools as st
import evoVAE.utils.metrics as mt
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAE import SeqVAE
import yaml
import evoVAE.utils.visualisation as vs 
from sklearn.linear_model import LassoLars, Ridge
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def get_vae_latent(model, loader, orig_data):
    
    latent = vs.get_mu(model, loader)
    latent.rename(columns={"id": "mutant"}, inplace=True)
    latent = latent.merge(orig_data[["mutant", "DMS_score", "DMS_score_bin"]], on="mutant")

    return latent

def create_vae_data(model, loader, orig_data):

    latent = get_vae_latent(model, loader, orig_data)    
    Xs_train = np.stack(latent["mu"])

    ys_train = np.array([float(y) for y in latent["DMS_score"]])
    ys_train_bin = np.array([int(y) for y in latent["DMS_score_bin"]])


    return Xs_train, ys_train, ys_train_bin


def train_and_fit_vae_top_model(train_data, test_data, state, label, protein, cls_list, param_grid_list, pipe, settings):
    
    device = torch.device("cpu")
    train_loader = setup_loader(train_data, device)
    test_loader = setup_loader(test_data, device)

    print(label)
    seq_len = len(train_data["mutated_sequence"].values[0])
    model = instantiate_model(seq_len, device, state, settings)

    Xs_train, ys_train, ys_train_bin = create_vae_data(model, train_loader, train_data)
    Xs_test,  ys_test, ys_test_bin = create_vae_data(model, test_loader, test_data)

    results, grids = perform_grid_search(Xs_train, ys_train, cls_list, param_grid_list, pipe)

    spear = {}
    for grid in grids:
        model_name = str(grid.best_estimator_.get_params()['model']).split("(")[0]
        model_alpha = grid.best_estimator_.get_params()["model__alpha"]
        spear[model_name] = {}
        spear[model_name]["alpha"] = model_alpha

        # TEST DATA
        preds = grid.predict(Xs_test)
        spearmanr = scipy.stats.spearmanr(ys_test, preds).statistic
        k_recall = mt.top_k_recall(preds, ys_test)
        ndcg = mt.calc_ndcg(ys_test, preds)
        roc_auc = roc_auc_score(ys_test_bin, preds)
        spear[model_name]["test_spearmanr"] = spearmanr
        spear[model_name]["test_k_recall"] = k_recall
        spear[model_name]["test_ndcg"] = ndcg
        spear[model_name]["test_roc_auc"] = roc_auc

        # TRAIN DATA
        preds = grid.predict(Xs_train)
        spearmanr = scipy.stats.spearmanr(ys_train, preds).statistic
        k_recall = mt.top_k_recall(preds, ys_train)
        ndcg = mt.calc_ndcg(ys_train, preds)
        roc_auc = roc_auc_score(ys_train_bin, preds)
        spear[model_name]["train_spearmanr"] = spearmanr
        spear[model_name]["train_k_recall"] = k_recall
        spear[model_name]["train_ndcg"] = ndcg
        spear[model_name]["train_roc_auc"] = roc_auc
        
    return (results, grids, spear)

def instantiate_top_models_and_pipe():
    ridge_grid = [
        {
            'model': [Ridge()],
            'model__alpha': np.logspace(-6, 6, 1000),
            'model__solver': ["auto"]
        }
    ]

    lasso_grid = [
        {
            'model': [LassoLars()],
            'model__alpha': np.logspace(-6, 6, 1000),
        }
    ]

    cls_list = [Ridge, LassoLars]
    param_grid_list = [ridge_grid, lasso_grid]

    pipe = Pipeline(
        steps = [
            ('model', 'passthrough')
        ]
    )

    return cls_list, param_grid_list, pipe

def perform_grid_search(train_data, test_data, classifier_classes: list, 
                        grid_parameters: list, pipeline: Pipeline, cv=10) -> Tuple[list, list]:
    result_list = []
    grid_list = []
    for cls_name, param_grid in zip(classifier_classes, grid_parameters):
        print(cls_name)
        grid = GridSearchCV(
            estimator = pipeline,
            param_grid = param_grid,
            scoring = 'r2',
            verbose = 0,
            cv=cv,
            n_jobs = -1, # use all available cores
            return_train_score=True,
        )

        grid.fit(train_data, test_data)
        result_list.append(pd.DataFrame.from_dict(grid.cv_results_))
        grid_list.append(grid)

    return result_list, grid_list,

def setup_loader(data, device):
    
    dataset = MSA_Dataset(
        data["one_hot"],
        np.arange(len(data["one_hot"])),
        data["mutant"],
        device=device,      
    )
    
    print(f"Dataset size: {len(dataset)}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return loader

def instantiate_model(seq_len, device, state_dict, settings):
    
    model = SeqVAE(dim_latent_vars=settings["latent_dims"],
                dim_msa_vars= (seq_len * 21), 
                num_hidden_units=settings["hidden_dims"],
                settings=settings,
                num_aa_type=settings["AA_count"])

    model = model.to(device)
    model.load_state_dict(torch.load(state_dict, map_location=device))
    return model 

if __name__ == "__main__":
    
    protein = ["a4", "gcn4", "gfp", "mafg", "gb1"]
    data = ["ae", "a", "e"]
    labels = ["Ancestor/Extant", "Ancestor", "Extant"]
    latent_dims = [7, 5, 3, 4, 8]

    datasets = ["A4_HUMAN_Seuma_2022.csv",
                "GCN4_YEAST_Staller_2018.csv",
                "GFP_AEQVI_Sarkisyan_2016.csv",
                "MAFG_MOUSE_Tsuboyama_2023_1K1V.csv", 
                "SPG1_STRSG_Wu_2016.csv"]

    RIDGE = 0
    LASSO = 1

    final = [["model", "family", "category", "train_set", "spearman", "k_recall", "ndcg", "roc_auc"]]
    for p, v, dims in zip(protein, datasets, latent_dims):

        variants = pd.read_csv(v)
        variants["one_hot"] = variants["mutated_sequence"].apply(st.seq_to_one_hot)
        train, test = train_test_split(variants, train_size=0.8, random_state=42)
        
        for d, lab, in zip(data, labels):
            
            with open("dummy_config.yaml", "r") as stream:
                settings = yaml.safe_load(stream)
            settings["latent_dims"] = dims
            state_dict = f"{p}_{d}_r1_model_state.pt"
            cls_list, param_grid_list, pipe = instantiate_top_models_and_pipe()
            results, grids, spearman = train_and_fit_vae_top_model(train, test, state_dict, lab, p,
                                                        cls_list, param_grid_list, pipe, settings)
            
            models = ["Ridge", "LassoLars"]
            training = ["train", "test"]
            for model in models:
                for t in training:
                    final.append([model, p, lab, t, spearman[model][f"{t}_spearmanr"], spearman[model][f"{t}_k_recall"], 
                                spearman[model][f"{t}_ndcg"], spearman[model][f"{t}_roc_auc"]])
                
    df = pd.DataFrame(final[1:], columns=final[0])
    df.to_csv("vae_top_model_spearman.csv", index=False)
    
