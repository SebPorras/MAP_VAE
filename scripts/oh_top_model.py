"""
oh_top_model.py

This code is used to train a Ridge and LARS lasso regression to predict DMS fitness 
using one-hot encodings of sequences. 

Be careful with n_jobs=-1 in GridSearchCV as it will use all available cores.
"""

import torch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import scipy
import MAP_VAE.utils.seq_tools as st
import MAP_VAE.utils.metrics as mt

import yaml
import MAP_VAE.utils.visualisation as vs
from sklearn.linear_model import LassoLars, Ridge
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def instantiate_top_models_and_pipe():
    ridge_grid = [
        {
            "model": [Ridge()],
            "model__alpha": np.logspace(-6, 6, 1000),
            "model__solver": ["auto"],
        }
    ]

    lasso_grid = [
        {
            "model": [LassoLars()],
            "model__alpha": np.logspace(-6, 6, 1000),
        }
    ]

    cls_list = [Ridge, LassoLars]
    param_grid_list = [ridge_grid, lasso_grid]

    pipe = Pipeline(steps=[("model", "passthrough")])

    return cls_list, param_grid_list, pipe


def perform_grid_search(
    train_data,
    test_data,
    classifier_classes: list,
    grid_parameters: list,
    pipeline: Pipeline,
    cv=5,
) -> Tuple[list, list]:
    result_list = []
    grid_list = []
    for cls_name, param_grid in zip(classifier_classes, grid_parameters):
        print(cls_name)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="r2",
            verbose=0,
            cv=cv,
            n_jobs=-1,  # use all available cores
            return_train_score=True,
        )

        grid.fit(train_data, test_data)
        result_list.append(pd.DataFrame.from_dict(grid.cv_results_))
        grid_list.append(grid)

    return (
        result_list,
        grid_list,
    )


def train_and_fit_top_model(train, test):

    xs_train = np.array(
        [st.seq_to_one_hot(x).flatten() for x in train["mutated_sequence"]]
    )
    ys_train = np.array([float(y) for y in train["DMS_score"]])
    ys_train_bin = np.array([float(y) for y in train["DMS_score_bin"]])

    xs_test = np.array(
        [st.seq_to_one_hot(x).flatten() for x in test["mutated_sequence"]]
    )
    ys_test = np.array([float(y) for y in test["DMS_score"]])
    ys_test_bin = np.array([float(y) for y in test["DMS_score_bin"]])

    cls_list, param_grid_list, pipe = instantiate_top_models_and_pipe()
    results, grids = perform_grid_search(
        xs_train, ys_train, cls_list, param_grid_list, pipe
    )

    results = {}
    for grid in grids:
        model_name = str(grid.best_estimator_.get_params()["model"]).split("(")[0]
        model_alpha = grid.best_estimator_.get_params()["model__alpha"]
        results[model_name] = {}
        results[model_name]["alpha"] = model_alpha

        # TEST DATA
        preds = grid.predict(xs_test)
        spearmanr = scipy.stats.spearmanr(ys_test, preds).statistic
        k_recall = mt.top_k_recall(preds, ys_test)
        ndcg = mt.calc_ndcg(ys_test, preds)
        roc_auc = roc_auc_score(ys_test_bin, preds)
        results[model_name]["test_spearmanr"] = spearmanr
        results[model_name]["test_k_recall"] = k_recall
        results[model_name]["test_ndcg"] = ndcg
        results[model_name]["test_roc_auc"] = roc_auc

        # TRAIN DATA
        preds = grid.predict(xs_train)
        spearmanr = scipy.stats.spearmanr(ys_train, preds).statistic
        k_recall = mt.top_k_recall(preds, ys_train)
        ndcg = mt.calc_ndcg(ys_train, preds)
        roc_auc = roc_auc_score(ys_train_bin, preds)
        results[model_name]["train_spearmanr"] = spearmanr
        results[model_name]["train_k_recall"] = k_recall
        results[model_name]["train_ndcg"] = ndcg
        results[model_name]["train_roc_auc"] = roc_auc

    return results


if __name__ == "__main__":

    # protein = ["a4", "gcn4", "gfp", "mafg", "gb1"]
    # datasets = ["A4_HUMAN_Seuma_2022.csv",
    #             "GCN4_YEAST_Staller_2018.csv",
    #             "GFP_AEQVI_Sarkisyan_2016.csv",
    #             "MAFG_MOUSE_Tsuboyama_2023_1K1V.csv",
    #             "SPG1_STRSG_Wu_2016.csv"]

    protein = ["gfp"]
    datasets = [
        "/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv"
    ]
    training = ["ancestors_extants", "ancestors", "extants"]

    RIDGE = 0
    LASSO = 1

    with open("gfp_one_hot_top_model_results.csv", "a") as f:
        # f.write("model,family,train_set,spearman,k_recall,ndcg,roc_auc\n")
        for p, v in zip(protein, datasets):
            for train in training:

                variants = pd.read_csv(v)
                train, test = train_test_split(
                    variants, train_size=0.8, random_state=42
                )

                results = train_and_fit_top_model(train, test)

                models = ["Ridge", "LassoLars"]
                training = ["train", "test"]
                for model in models:
                    for t in training:
                        f.write(
                            f"{model},{p},{t},{results[model][f'{t}_spearmanr']},{results[model][f'{t}_k_recall']},"
                            f"{results[model][f'{t}_ndcg']},{results[model][f'{t}_roc_auc']}\n"
                        )
                        f.flush()
