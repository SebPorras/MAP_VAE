# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +

import evoVAE.utils.visualisation as vs
import evoVAE.utils.seq_tools as st
import yaml
#pd.set_option("display.max_rows", None)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import torch 
import numpy as np
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAE import SeqVAE
# -

# This notebook handles visualisation of the VAE latent space. 
#
# Instantiates a model from a dummy config, the most important thing is that the number of layers and latent space dimensions matches what you used for training, other hyperparameters in the config file do not matter. 
#
# There a visualisations that shows how the model percieves the training data after training. 
#
# There is a visualisation for a single tree and there is also functionality to write these 3D coordinates out to a file

# # Model init

with open("../data/dummy_config.yaml", "r") as stream:
    settings = yaml.safe_load(stream)


# # Visualising training data
#
# This is going to need to be done on a GPU, can't run it on my local device. 

# # GFP 

# #### Visualise the latent space

path = "/Users/sebs_mac/uni_OneDrive/honours/data/gfp/independent_runs/no_synthetic/alns/"
gfp_ae_latent, gfp_a_latent, gfp_e_latent = vs.get_model_embeddings(path, 
                                                                 ae_file_name="gfp_ancestors_extants_no_syn_no_dupes.pkl",
                                                                 variant_path="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv",
                                                                 ae_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/gfp_standard/gfp_ae/gfp_ae_r9/gfp_ae_r9_model_state.pt",
                                                                 a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/gfp_standard/gfp_a/gfp_a_r4/gfp_a_r4_model_state.pt",
                                                                 e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/gfp_standard/gfp_e/gfp_e_r10/gfp_e_r10_model_state.pt",
                                                                 settings=settings)

# +
model_reps = [gfp_ae_latent, gfp_a_latent, gfp_e_latent]
labels = ["AE model", "A model", "E model"]

fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': '3d'})

for rep, label, ax in zip(model_reps, labels, axes):
    ax.set_title("GFP: " + label)
    vs.tree_vis_3d(rep, "GFP_AEQVI/1-238", rgb=True, ax=ax)

  
#plt.tight_layout()
plt.savefig("gfp_3D_rgb.png", dpi=300)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for rep, label, ax in zip(model_reps, labels, axes):
    ax.set_title("GFP: " + label)
    vs.tree_vis_2d(rep, "GFP_AEQVI/1-238", rgb=True, ax=ax)

plt.tight_layout()
#plt.savefig("gfp_2D_rgb.png", dpi=300)
plt.show()


# -

# #### GFP - convert latent space coordinates to ITOL annotations 

# +

path = "/Users/sebs_mac/uni_OneDrive/honours/data/gfp/independent_runs/no_synthetic/ancestors/auto_rooted/ancestors/"
gfp_tree = path + "run_1_ancestors_extants.fa"

vs.latent_tree_to_itol("gfp", 
                 tree_seq_path=gfp_tree, 
                 ae_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/gfp_standard/gfp_ae/gfp_ae_r9/gfp_ae_r9_model_state.pt",
                 a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/gfp_standard/gfp_a/gfp_a_r4/gfp_a_r4_model_state.pt",
                 e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/gfp_standard/gfp_e/gfp_e_r10/gfp_e_r10_model_state.pt",
                 settings=settings
                 )
# -

# # GB1

# #### Visualise the latent space

path = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/alns/"
gb1_ae_latent, gb1_a_latent, gb1_e_latent = vs.get_model_embeddings(path, 
                                                                 ae_file_name="gb1_ancestors_extants_no_dupes.pkl",
                                                                 variant_path="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/SPG1_STRSG_Wu_2016.csv",
                                                                 ae_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gb1_standard/gb1_ae/gb1_ae_r1/gb1_ae_r1_model_state.pt",
                                                                 a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gb1_standard/gb1_a/gb1_a_r2/gb1_a_r2_model_state.pt",
                                                                 e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gb1_standard/gb1_e/gb1_e_r2/gb1_e_r2_model_state.pt",
                                                                 settings=settings)

gb1_ae_latent.tail()

# +

model_reps = [gb1_ae_latent, gb1_a_latent, gb1_e_latent]
labels = ["AE model", "A model", "E model"]

fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': '3d'})

for rep, label, ax in zip(model_reps, labels, axes):
    tree_vis_3d(rep, "SPG1_STRSG/1-448", rgb=True, ax=ax)
    ax.set_title("GB1: " + label)


#plt.tight_layout()
#plt.savefig("gb1_3D.png", dpi=300)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for rep, label, ax in zip(model_reps, labels, axes):
    tree_vis_2d(rep, "SPG1_STRSG/1-448", rgb=True, ax=ax)
    ax.set_title("GB1: " + label)

plt.tight_layout()
#plt.savefig("gb1_2D_rgb.png", dpi=300)
plt.show()


# -

# #### ITOL annotations

# +
path = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/ancestors/"
gb1_tree = path + "anc_tree_1_ancestors_extants.aln"

vs.latent_tree_to_itol("gb1", 
                 tree_seq_path=gb1_tree, 
                 ae_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/gb1_standard/gb1_ae/gb1_ae_r1/gb1_ae_r1_model_state.pt",
                 a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/gb1_standard/gb1_a/gb1_a_r2/gb1_a_r2_model_state.pt",
                 e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/gb1_standard/gb1_e/gb1_e_r2/gb1_e_r2_model_state.pt",
                 settings=settings
                 )
# -

# # A4

path = "/Users/sebs_mac/uni_OneDrive/honours/data/a4_human/alns/"
a4_ae_latent, a4_a_latent, a4_e_latent = vs.get_model_embeddings(path, 
                                                                 ae_file_name="a4_ancestors_extants_no_dupes.pkl",
                                                                 variant_path="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/A4_HUMAN_Seuma_2022.csv",
                                                                 ae_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/a4_standard/a4_ae/a4_ae_r2/a4_ae_r2_model_state.pt",
                                                                 a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/a4_standard/a4_a/a4_a_r9/a4_a_r9_model_state.pt",
                                                                 e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/a4_standard/a4_e/a4_e_r9/a4_e_r9_model_state.pt",
                                                                 settings=settings)



# +
model_reps = [a4_ae_latent, a4_a_latent, a4_e_latent]
labels = ["AE model", "A model", "E model"]

fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': '3d'})

for rep, label, ax in zip(model_reps, labels, axes):
    ax.set_title(label)
    vs.tree_vis_3d(rep, "A4_HUMAN/1-770", rgb=True, ax=ax)
    

#plt.tight_layout()
plt.savefig("a4_3D_rgb.png", dpi=300)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for rep, label, ax in zip(model_reps, labels, axes):
    ax.set_title(label)
    vs.tree_vis_2d(rep, "A4_HUMAN/1-770", rgb=True, ax=ax)


plt.tight_layout()
plt.savefig("a4_2D_rgb.png", dpi=300)
plt.show()

# +
path = "/Users/sebs_mac/uni_OneDrive/honours/data/a4_human/ancestors/"
a4_tree = path + "tree_1_ancestors_extants.aln"

vs.latent_tree_to_itol("a4", 
                 tree_seq_path=a4_tree, 
                 ae_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/a4_standard/a4_ae/a4_ae_r2/a4_ae_r2_model_state.pt",
                 a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/a4_standard/a4_a/a4_a_r9/a4_a_r9_model_state.pt",
                 e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/a4_standard/a4_e/a4_e_r9/a4_e_r9_model_state.pt",
                 settings=settings
                 )
# -

# # GCN4

path = "/Users/sebs_mac/uni_OneDrive/honours/data/gcn4/alns/"
gcn4_ae_latent, gcn4_a_latent, gcn4_e_latent = vs.get_model_embeddings(path, 
                                                                 ae_file_name="gcn4_ancestors_extants_no_dupes.pkl",
                                                                 variant_path="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GCN4_YEAST_Staller_2018.csv",
                                                                 ae_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gcn4_standard/gcn4_ae/gcn4_ae_r8/gcn4_ae_r8_model_state.pt",
                                                                 a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gcn4_standard/gcn4_a/gcn4_a_r6/gcn4_a_r6_model_state.pt",
                                                                 e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gcn4_standard/gcn4_e/gcn4_e_r1/gcn4_e_r1_model_state.pt",
                                                                 settings=settings)


# +
model_reps = [gcn4_ae_latent, gcn4_a_latent, gcn4_e_latent]
labels = ["AE model", "A model", "E model"]

fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': '3d'})

for rep, label, ax in zip(model_reps, labels, axes):
    ax.set_title("GCN4: " + label)
    vs.tree_vis_3d(rep, "GCN4_YEAST/1-281", rgb=True, ax=ax)

#plt.tight_layout()
#plt.savefig("gcn4_3D_rgb.png", dpi=300)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for rep, label, ax in zip(model_reps, labels, axes):
    ax.set_title("GCN4: " + label)
    vs.tree_vis_2d(rep, "GCN4_YEAST/1-281", rgb=True, ax=ax)
   
plt.tight_layout()
#plt.savefig("gcn4_2D.png", dpi=300)
plt.show()

# +
path = "/Users/sebs_mac/uni_OneDrive/honours/data/gcn4/ancestors/"
gcn4_tree = path + "tree_0_ancestors_extants.aln"

vs.latent_tree_to_itol("gcn4", 
                 tree_seq_path=gcn4_tree, 
                 ae_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/gcn4_standard/gcn4_ae/gcn4_ae_r8/gcn4_ae_r8_model_state.pt",
                 a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/gcn4_standard/gcn4_a/gcn4_a_r6/gcn4_a_r6_model_state.pt",
                 e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/gcn4_standard/gcn4_e/gcn4_e_r1/gcn4_e_r1_model_state.pt",
                 settings=settings
                 )
# -

# # MAFG 

path = "/Users/sebs_mac/uni_OneDrive/honours/data/mafg_mouse/alns/"
mafg_ae_latent, mafg_a_latent, mafg_e_latent = vs.get_model_embeddings(path, 
                                                                 ae_file_name="mafg_ancestors_extants_no_dupes.pkl",
                                                                 variant_path="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/MAFG_MOUSE_Tsuboyama_2023_1K1V.csv",
                                                                 ae_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/mafg_standard/mafg_ae/mafg_ae_r1/mafg_ae_r1_model_state.pt",
                                                                 a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/mafg_standard/mafg_a/mafg_a_r1/mafg_a_r1_model_state.pt",
                                                                 e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/mafg_standard/mafg_e/mafg_e_r1/mafg_e_r1_model_state.pt",
                                                                 settings=settings)


# +
model_reps = [mafg_ae_latent, mafg_a_latent, mafg_e_latent]
labels = ["AE model", "A model", "E model"]

fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': '3d'})

for rep, label, ax in zip(model_reps, labels, axes):
    vs.plot_latent_3D(rep, "MAFG_MOUSE/1-41", label, ax, "MAFG", rgb=False,ext_frac=0.5, anc_frac=0.3, var_frac=1)

  
#plt.tight_layout()
plt.savefig("mafg_3D.png", dpi=300)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for rep, label, ax in zip(model_reps, labels, axes):
    vs.plot_latent_2D(rep, "MAFG_MOUSE/1-41", label, ax, "MAFG", rgb=False, var_frac=1)

plt.tight_layout()
plt.savefig("mafg_2D.png", dpi=300)
plt.show()

# +
path = "/Users/sebs_mac/uni_OneDrive/honours/data/mafg_mouse/ancestors/"
mafg_tree = path + "tree_0_ancestors_extants.aln"

vs.latent_tree_to_itol("mafg", 
                 tree_seq_path=mafg_tree, 
                 ae_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/mafg_standard/mafg_ae/mafg_ae_r1/mafg_ae_r1_model_state.pt",
                 a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/mafg_standard/mafg_a/mafg_a_r1/mafg_a_r1_model_state.pt",
                 e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/mafg_standard/mafg_e/mafg_e_r1/mafg_e_r1_model_state.pt",
                 settings=settings
                 )
# -

# # Visualising single trees

# ## Cassowary RNAseZ

# +
path = "/Users/sebs_mac/uni_OneDrive/honours/data/cassowary/vis/"
cass_tree = path + "tree_1_ancestors_extants.fasta"
a_state_dict=  "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/cassowary_standard/cassowary_a_r1/cassowary_a_r1_model_state.pt"
e_state_dict = "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/cassowary_standard/cassowary_e_r1/cassowary_e_r1_model_state.pt"


vs.vis_tree(None, cass_tree, a_state_dict, settings, "RNAseZ - Ancestor model", rgb=True, lower_2d=True)
vs.vis_tree(None, cass_tree, e_state_dict, settings, "RNAseZ - Extant model", rgb=True, lower_2d=True)
vs.vis_tree(None, cass_tree, a_state_dict, settings, "RNAseZ - Ancestor model", rgb=True)
vs.vis_tree(None, cass_tree, e_state_dict, settings, "RNAseZ - Extant model", rgb=True)
# -

# ## GB1

# +
path = "/Users/sebs_mac/uni_OneDrive/honours/data/gb1/ancestors/"
gb1_tree = path + "anc_tree_1_ancestors_extants.aln"
a_state_dict=  "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gb1_standard/gb1_a/gb1_a_r1/gb1_a_r1_model_state.pt"
e_state_dict = "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gb1_standard/gb1_e/gb1_e_r1/gb1_e_r1_model_state.pt"

vs.vis_tree("SPG1_STRSG/1-448", gb1_tree, a_state_dict, settings, "GB1 - Ancestor model", rgb=True, lower_2d=True)
vs.vis_tree("SPG1_STRSG/1-448", gb1_tree, e_state_dict, settings, "GB1 - Extant model", rgb=True, lower_2d=True)
vs.vis_tree("SPG1_STRSG/1-448", gb1_tree, a_state_dict, settings, "GB1 - Ancestor model", rgb=True)
vs.vis_tree("SPG1_STRSG/1-448", gb1_tree, e_state_dict, settings, "GB1 - Extant model", rgb=True)
# -

# ## MAFG

# +
path = "/Users/sebs_mac/uni_OneDrive/honours/data/mafg_mouse/ancestors/"
mafg_tree = path + "tree_0_ancestors_extants.aln"
a_state_dict=  "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/mafg_standard/mafg_a/mafg_a_r1/mafg_a_r1_model_state.pt"
e_state_dict = "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/mafg_standard/mafg_e/mafg_e_r1/mafg_e_r1_model_state.pt"


vs.vis_tree("MAFG_MOUSE/1-41", mafg_tree, a_state_dict, settings, "MAFG - Ancestor model", rgb=True, lower_2d=True)
vs.vis_tree("MAFG_MOUSE/1-41", mafg_tree, e_state_dict, settings, "MAFG - Extant model", rgb=True, lower_2d=True)
vs.vis_tree("MAFG_MOUSE/1-41", mafg_tree, a_state_dict, settings, "MAFG - Ancestor model", rgb=True)
vs.vis_tree("MAFG_MOUSE/1-41", mafg_tree, e_state_dict, settings, "MAFG - Extant model", rgb=True)


# +

path = "/Users/sebs_mac/uni_OneDrive/honours/data/gcn4/ancestors/"
gcn4_tree = path + "tree_0_ancestors_extants.aln"

a_state_dict=  "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gcn4_standard/gcn4_a/gcn4_a_r1/gcn4_a_r1_model_state.pt"
e_state_dict = "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gcn4_standard/gcn4_e/gcn4_e_r1/gcn4_e_r1_model_state.pt"


vs.vis_tree("GCN4_YEAST/1-281", gcn4_tree, a_state_dict, settings, "GCN4 - Ancestor model", rgb=True, lower_2d=True)
vs.vis_tree("GCN4_YEAST/1-281", gcn4_tree, e_state_dict, settings, "GCN4 - Extant model", rgb=True, lower_2d=True)
vs.vis_tree("GCN4_YEAST/1-281", gcn4_tree, a_state_dict, settings, "GCN4 - Ancestor model", rgb=True)
vs.vis_tree("GCN4_YEAST/1-281", gcn4_tree, e_state_dict, settings, "GCN4 - Extant model", rgb=True)


# +

path = "/Users/sebs_mac/uni_OneDrive/honours/data/a4_human/ancestors/"
a4_tree = path + "tree_1_ancestors_extants.aln"

a_state_dict=  "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/a4_standard/a4_a/a4_a_r1/a4_a_r1_model_state.pt"
e_state_dict = "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/a4_standard/a4_e/a4_e_r1/a4_e_r1_model_state.pt"

vs.vis_tree("A4_HUMAN/1-770", a4_tree, a_state_dict, settings, "A4 - Ancestor model", rgb=True, lower_2d=True)
vs.vis_tree("A4_HUMAN/1-770", a4_tree, e_state_dict, settings, "A4 - Extant model", rgb=True, lower_2d=True)
vs.vis_tree("A4_HUMAN/1-770", a4_tree, a_state_dict, settings,  "A4 - Ancestor model", rgb=True)
vs.vis_tree("A4_HUMAN/1-770", a4_tree, e_state_dict, settings, "A4 - Extant model", rgb=True)

# -

# ### GFP

# +

path = "/Users/sebs_mac/uni_OneDrive/honours/data/gfp/independent_runs/no_synthetic/vis/"
gfp_tree = path + "tree_1_ancestors_extants.aln"

a_state_dict=  "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gfp_standard/gfp_a/gfp_a_r4/gfp_a_r4_model_state.pt"
e_state_dict = "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gfp_standard/gfp_e/gfp_e_r10/gfp_e_r10_model_state.pt"

vs.vis_tree("GFP_AEQVI/1-238", gfp_tree, a_state_dict, settings, "GFP - Ancestor model", rgb=True, lower_2d=True)
vs.vis_tree("GFP_AEQVI/1-238", gfp_tree, e_state_dict, settings, "GFP - Extant model", rgb=True, lower_2d=True)
vs.vis_tree("GFP_AEQVI/1-238", gfp_tree, a_state_dict, settings, "GFP - Ancestor model", rgb=True)
vs.vis_tree("GFP_AEQVI/1-238", gfp_tree, e_state_dict, settings, "GFP - Extant model", rgb=True)


# +
path = "/Users/sebs_mac/uni_OneDrive/honours/data/gfp/independent_runs/no_synthetic/ancestors/auto_rooted/ancestors/"
gfp_tree = path + "run_1_ancestors_extants.fa"

vs.latent_tree_to_itol("gfp", 
                 tree_seq_path=gfp_tree, 
                 ae_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gfp_standard/gfp_ae/gfp_ae_r9/gfp_ae_r9_model_state.pt",
                 a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gfp_standard/gfp_a/gfp_a_r4/gfp_a_r4_model_state.pt",
                 e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gfp_standard/gfp_e/gfp_e_r10/gfp_e_r10_model_state.pt",
                 settings=settings
                 )


# -

# # Visualise variants

# +

def vis_variants_3d(model, state_dict, device, title, variant_data, variant_loader):

    model.load_state_dict(torch.load(state_dict, map_location=device))

    id_to_mu = vs.get_mu(model, variant_loader)
    id_to_mu.rename(columns={"id": "mutant"}, inplace=True)
    merged = variant_data.merge(id_to_mu, on="mutant")

    variant_mus = np.stack(merged["mu"])
    #scaler = MinMaxScaler()
    #dms_values = scaler.fit_transform(merged["DMS_score"].values.reshape(-1, 1))


    fig, (ax) = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={"projection": "3d"})
    scatter = ax.scatter(variant_mus[:, 0], variant_mus[:, 1], variant_mus[:, 2], c=merged["DMS_score"], cmap='viridis')

    ax.set_title(title)
    ax.set_xlabel("Z1")
    ax.set_ylabel("Z2")
    ax.set_zlabel("Z3")
    # Add color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Fitness Score')

    plt.show()


def vis_variants_2d(model, state_dict, device, title, variant_data, variant_loader):

    model.load_state_dict(torch.load(state_dict, map_location=device))

    id_to_mu = vs.get_mu(model, variant_loader)
    id_to_mu.rename(columns={"id": "mutant"}, inplace=True)
    merged = variant_data.merge(id_to_mu, on="mutant")

    pca = PCA(n_components=2)

    zs_2d = pca.fit_transform(np.vstack(merged["mu"].values))
    merged["pca"] = list(zs_2d)

    #scaler = MinMaxScaler()
    #dms_values = scaler.fit_transform(merged["DMS_score"].values.reshape(-1, 1))

    fig, (ax) = plt.subplots(1, 1, figsize=(12, 8))
    scatter = ax.scatter(
        np.vstack(merged["pca"].values)[:, 0],
        np.vstack(merged["pca"].values)[:, 1],
        c=merged["DMS_score"],

    )

    ax.set_title(title)
    ax.set_xlabel(f"PCA1 ({round(pca.explained_variance_[0] * 100, 2)}%)")
    ax.set_ylabel(f"PCA2 ({round(pca.explained_variance_[1] * 100, 2)}%)")

    # Add color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Fitness Score')

    plt.show()

with open("../data/dummy_config.yaml", "r") as stream:
    settings = yaml.safe_load(stream)
# -

# ### GB1

a_state_dict=  "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gb1_standard/gb1_a/gb1_a_r1/gb1_a_r1_model_state.pt"
e_state_dict = "/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gb1_standard/gb1_e/gb1_e_r1/gb1_e_r1_model_state.pt"
variant_path="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/SPG1_STRSG_Wu_2016.csv"

# +
variants = pd.read_csv(variant_path)

variants["encoding"] = variants["mutated_sequence"].apply(st.seq_to_one_hot)

num_seqs = len(variants["mutated_sequence"])
device = torch.device("cpu")

var_dataset = MSA_Dataset(
    variants["encoding"],
    np.arange(len(variants["encoding"])),
    variants["mutant"],
    device=device,
)

loader = torch.utils.data.DataLoader(var_dataset, batch_size=num_seqs, shuffle=False)
seq_len = var_dataset[0][0].shape[0]
print(seq_len)
input_dims = seq_len * settings["AA_count"]

model = SeqVAE(
    dim_latent_vars=settings["latent_dims"],
    dim_msa_vars=input_dims,
    num_hidden_units=settings["hidden_dims"],
    settings=settings,
    num_aa_type=settings["AA_count"],
)
model = model.to(device)

# +

vis_variants_3d(model, state_dict=a_state_dict, device=device, 
                title="GB1: Ancestor model - Variants", variant_data=variants, variant_loader=loader)


vis_variants_3d(model, state_dict=e_state_dict, device=device, 
                title="GB1: Extant model - Variants", variant_data=variants, variant_loader=loader)

# -

#

# +
vis_variants_2d(model, state_dict=a_state_dict, device=device, 
                title="GB1: Ancestor model - Variants", variant_data=variants, variant_loader=loader)


vis_variants_2d(model, state_dict=e_state_dict, device=device, 
                title="GB1: Extant model - Variants", variant_data=variants, variant_loader=loader)

# -

# ### A4 

variant_path="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/A4_HUMAN_Seuma_2022.csv"
a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/a4_standard/a4_a/a4_a_r9/a4_a_r9_model_state.pt"
e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/a4_standard/a4_e/a4_e_r9/a4_e_r9_model_state.pt"

# +
variants = pd.read_csv(variant_path)

variants["encoding"] = variants["mutated_sequence"].apply(st.seq_to_one_hot)

num_seqs = len(variants["mutated_sequence"])
device = torch.device("cpu")

var_dataset = MSA_Dataset(
    variants["encoding"],
    np.arange(len(variants["encoding"])),
    variants["mutant"],
    device=device,
)

loader = torch.utils.data.DataLoader(var_dataset, batch_size=num_seqs, shuffle=False)
seq_len = var_dataset[0][0].shape[0]
print(seq_len)
input_dims = seq_len * settings["AA_count"]

model = SeqVAE(
    dim_latent_vars=settings["latent_dims"],
    dim_msa_vars=input_dims,
    num_hidden_units=settings["hidden_dims"],
    settings=settings,
    num_aa_type=settings["AA_count"],
)
model = model.to(device)



# +


vis_variants_3d(model, state_dict=a_state_dict, device=device, 
                title="A4: Ancestor model - Variants", variant_data=variants, variant_loader=loader)


vis_variants_3d(model, state_dict=e_state_dict, device=device, 
                title="A4: Extant model - Variants", variant_data=variants, variant_loader=loader)


# +

vis_variants_2d(model, state_dict=a_state_dict, device=device, 
                title="A4: Ancestor model - Variants", variant_data=variants, variant_loader=loader)


vis_variants_2d(model, state_dict=e_state_dict, device=device, 
                title="A4: Extant model - Variants", variant_data=variants, variant_loader=loader)

# -

# ### MAFG

variant_path="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/MAFG_MOUSE_Tsuboyama_2023_1K1V.csv"
a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/mafg_standard/mafg_a/mafg_a_r1/mafg_a_r1_model_state.pt"
e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/mafg_standard/mafg_e/mafg_e_r1/mafg_e_r1_model_state.pt"

# +
variants = pd.read_csv(variant_path)

variants["encoding"] = variants["mutated_sequence"].apply(st.seq_to_one_hot)

num_seqs = len(variants["mutated_sequence"])
device = torch.device("cpu")

var_dataset = MSA_Dataset(
    variants["encoding"],
    np.arange(len(variants["encoding"])),
    variants["mutant"],
    device=device,
)

loader = torch.utils.data.DataLoader(var_dataset, batch_size=num_seqs, shuffle=False)
seq_len = var_dataset[0][0].shape[0]
print(seq_len)
input_dims = seq_len * settings["AA_count"]

model = SeqVAE(
    dim_latent_vars=settings["latent_dims"],
    dim_msa_vars=input_dims,
    num_hidden_units=settings["hidden_dims"],
    settings=settings,
    num_aa_type=settings["AA_count"],
)
model = model.to(device)

# +

vis_variants_3d(model, state_dict=a_state_dict, device=device, 
                title="MAFG: Ancestor model - Variants", variant_data=variants, variant_loader=loader)


vis_variants_3d(model, state_dict=e_state_dict, device=device, 
                title="MAFG: Extant model - Variants", variant_data=variants, variant_loader=loader)


# +

vis_variants_2d(model, state_dict=a_state_dict, device=device, 
                title="MAFG: Ancestor model - Variants", variant_data=variants, variant_loader=loader)


vis_variants_2d(model, state_dict=e_state_dict, device=device, 
                title="MAFG: Extant model - Variants", variant_data=variants, variant_loader=loader)

# -

# ### GFP

variant_path="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv"
a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gfp_standard/gfp_a/gfp_a_r4/gfp_a_r4_model_state.pt"
e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gfp_standard/gfp_e/gfp_e_r10/gfp_e_r10_model_state.pt"

# +
variants = pd.read_csv(variant_path)

variants["encoding"] = variants["mutated_sequence"].apply(st.seq_to_one_hot)

num_seqs = len(variants["mutated_sequence"])
device = torch.device("cpu")

var_dataset = MSA_Dataset(
    variants["encoding"],
    np.arange(len(variants["encoding"])),
    variants["mutant"],
    device=device,
)

loader = torch.utils.data.DataLoader(var_dataset, batch_size=num_seqs, shuffle=False)
seq_len = var_dataset[0][0].shape[0]
print(seq_len)
input_dims = seq_len * settings["AA_count"]

model = SeqVAE(
    dim_latent_vars=settings["latent_dims"],
    dim_msa_vars=input_dims,
    num_hidden_units=settings["hidden_dims"],
    settings=settings,
    num_aa_type=settings["AA_count"],
)
model = model.to(device)

# +

vis_variants_3d(model, state_dict=a_state_dict, device=device, 
                title="GFP: Ancestor model - Variants", variant_data=variants, variant_loader=loader)


vis_variants_3d(model, state_dict=e_state_dict, device=device, 
                title="GFP: Extant model - Variants", variant_data=variants, variant_loader=loader)


# +

vis_variants_2d(model, state_dict=a_state_dict, device=device, 
                title="GFP: Ancestor model - Variants", variant_data=variants, variant_loader=loader)


vis_variants_2d(model, state_dict=e_state_dict, device=device, 
                title="GFP: Extant model - Variants", variant_data=variants, variant_loader=loader)

# -

# ### GCN4

variant_path="/Users/sebs_mac/git_repos/dms_data/DMS_ProteinGym_substitutions/GCN4_YEAST_Staller_2018.csv"
a_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gcn4_standard/gcn4_a/gcn4_a_r6/gcn4_a_r6_model_state.pt"
e_state_dict="/Users/sebs_mac/uni_OneDrive/honours/data/standard_test_results/raw_data/gcn4_standard/gcn4_e/gcn4_e_r1/gcn4_e_r1_model_state.pt"

# +
variants = pd.read_csv(variant_path)

variants["encoding"] = variants["mutated_sequence"].apply(st.seq_to_one_hot)

num_seqs = len(variants["mutated_sequence"])
device = torch.device("cpu")

var_dataset = MSA_Dataset(
    variants["encoding"],
    np.arange(len(variants["encoding"])),
    variants["mutant"],
    device=device,
)

loader = torch.utils.data.DataLoader(var_dataset, batch_size=num_seqs, shuffle=False)
seq_len = var_dataset[0][0].shape[0]
print(seq_len)
input_dims = seq_len * settings["AA_count"]

model = SeqVAE(
    dim_latent_vars=settings["latent_dims"],
    dim_msa_vars=input_dims,
    num_hidden_units=settings["hidden_dims"],
    settings=settings,
    num_aa_type=settings["AA_count"],
)
model = model.to(device)

# +

vis_variants_3d(model, state_dict=a_state_dict, device=device, 
                title="GCN4: Ancestor model - Variants", variant_data=variants, variant_loader=loader)


vis_variants_3d(model, state_dict=e_state_dict, device=device, 
                title="GCN4: Extant model - Variants", variant_data=variants, variant_loader=loader)

# -

# # NR vis

# +
import torch 
from evoVAE.utils.datasets import MSA_Dataset
from evoVAE.models.seqVAE import SeqVAE
import yaml
import evoVAE.utils.visualisation as vs
import matplotlib.pyplot as plt
import seaborn as sns

path = "/Users/sebs_mac/uni_OneDrive/honours/data/nr/"
state_dict = f"{path}nr_r1_model_state.pt"
tree_seq_path = f"{path}NR_MSA.aln"

wt = ""
with open("../data/dummy_config.yaml", "r") as stream:
    settings = yaml.safe_load(stream)


tree_seqs = st.read_aln_file(tree_seq_path)

one_hot = tree_seqs["sequence"].apply(st.seq_to_one_hot)
tree_seqs["encoding"] = one_hot
num_seqs = len(tree_seqs["sequence"])
device = torch.device("mps")

tree_dataset = MSA_Dataset(
    tree_seqs["encoding"],
    np.arange(len(one_hot)),
    tree_seqs["id"],
    device=device,
)

tree_loader = torch.utils.data.DataLoader(
    tree_dataset, batch_size=num_seqs, shuffle=False
)

seq_len = tree_dataset[0][0].shape[0]
input_dims = seq_len * settings["AA_count"]

model = SeqVAE(
    dim_latent_vars=settings["latent_dims"],
    dim_msa_vars=input_dims,
    num_hidden_units=settings["hidden_dims"],
    settings=settings,
    num_aa_type=settings["AA_count"],
)

model = model.to(device)

model.load_state_dict(torch.load(state_dict, map_location=device))
latent = vs.get_mu(model, tree_loader)



# +
zs = np.array([z for z in latent["mu"]])

min_r = np.min(np.stack(latent["mu"])[:, 0])
max_r = np.max(np.stack(latent["mu"])[:, 0])
min_g = np.min(np.stack(latent["mu"])[:, 1])
max_g = np.max(np.stack(latent["mu"])[:, 1])
min_b = np.min(np.stack(latent["mu"])[:, 2])
max_b = np.max(np.stack(latent["mu"])[:, 2])

an_rgb = [
            (
        
                (r - min_r) / (max_r - min_r),
                (g - min_g) / (max_g - min_g),
                (b - min_b) / (max_b - min_b),
            )
            for r, g, b in zip(
                np.stack(zs)[:, 0],
                np.stack(zs)[:, 1],
                np.stack(zs)[:, 2],
            )
        ]


an_rgb = [vs.rgb_to_hex_normalized(*x) for x in an_rgb]
fig, (ax) = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={"projection": "3d"})
ax.scatter(
    zs[:, 0],
    zs[:, 1],
    zs[:, 2],
    c=an_rgb,
)
latent["COLOR"] = an_rgb
latent.drop(columns=["mu"], inplace=True)

# +
#latent.to_csv("/Users/sebs_mac/uni_OneDrive/honours/data/nr/latent_colours_itol.csv")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Step 1: Extract the second part of the IDs
ids = latent["id"].str.split("_", expand=True)[1]

# Step 2: Get unique IDs
unique_ids = ids.unique()

# Step 3: Create a color palette with as many colors as unique IDs
palette = sns.color_palette("hsv", len(unique_ids))

# Step 4: Map each unique ID to a color
id_to_color = {id_: palette[i] for i, id_ in enumerate(unique_ids)}
id_to_color

# Optional: Convert RGB colors to HEX
id_to_color_hex = {id_: vs.rgb_to_hex_normalized(*color) for id_, color in id_to_color.items()}

# print(id_to_color_hex)
id_to_color_hex
# -

latent["COLOR"] = [id_to_color_hex[y] for y in latent["id"].apply(lambda x: x.split("_")[1])]

latent.to_csv("/Users/sebs_mac/uni_OneDrive/honours/data/nr/annot_colours_itol.csv")

vs.write_itol_dataset_symbol("nr_known_annots_itol", latent)
