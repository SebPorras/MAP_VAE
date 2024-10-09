# MAP-VAE: Multiplexed Ancestral Phylogeny Variational AutoEncoder

Multiplexed ancestral sequence reconstruction (mASR) (Matthews et al., 2023) is a new approach that seeks to use stronger biological priors by training machine learning models with predicted ancestral proteins. By sampling multiple phylogenetic topologies and selecting statistically equivalent trees, a diverse set of ancestral sequences with unique mutational backgrounds and indel information can be generated. Using a simple variational autoencoder (VAE) and five protein families adapted from the ProteinGym benchmarking database (Notin et al., 2023), we benchmarked how using ancestors affected the ability of a model to complete downstream tasks such as prediction of protein mutation effects as well as capturing epistasis to see what additional information could be extracted from ancestors.

The code in this repository contains a standard VAE model along with scripts used to train and evaluate models trained with ancestors. It also contains a number of helper functions used to process sequence data. 

## Installation

Activate your conda environment and then run. 

Install with `pip install -e .`


## Notebook Demos

There are [Jupyter](https://jupyter.org/) notebooks distributed as plaintext Python files, which can be converted/synced with [`jupytext`](https://github.com/mwouts/jupytext).

```bash
jupytext --sync ./notebooks/<file>.py
```

## Model and config file

The main model: `src/models/seqVAE.py`

An example config file: `data/example_config.yaml`

## Training scripts

#### Help for each script can be found by using -h flag

Standard training: `scripts/train_vae.py`

Hyperparameter tuning: `scripts/tuner.py`

Zero-shot prediction: `scripts/zero_shot.py`

k-fold cross validation: `scripts/k_fold_train_vae.py`

