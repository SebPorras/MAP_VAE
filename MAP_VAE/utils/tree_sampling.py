# tree_sampling.py

from ete3 import Tree
import random
from typing import Tuple, List
import pandas as pd


def sample_ancestor_nodes(
    rejection_threshold: float, tree: Tree, tree_run: int
) -> Tuple[List[str], int]:
    """Do a level order traversal of a tree, at each ancestor, only sample if a random number
    between 0 and 1 is below 1 - rejection_threshold.

    Returns:
    list of node names, the number of nodes sampled
    """

    sampled_nodes = []
    for node in tree.traverse():

        # we're only interested in sampling ancestors
        if node.is_leaf():
            continue

        if random.random() <= 1 - rejection_threshold:
            # pkl file format
            sampled_nodes.append(node.name + f"_tree_{tree_run}")

    return sampled_nodes, len(sampled_nodes)


def sample_ancestor_trees(
    tree_count: int,
    rejection_threshold: float,
    invalid_trees: List[int],
    tree_path: str,
    ancestor_pkl_path: str,
    ete_read_setting: int = 1,
):
    """
    Take a path to where all trees are stored. For each tree, sample the ancestor nodes, only taking ancestors and only sampling
    when a random number is less than 1 - rejection_threshold.

    Returns:
    A dataframe with all the sampled_ancestors.
    """

    sampled_names = []
    for iteration in range(1, tree_count + 1):

        if iteration in invalid_trees:
            continue

        # default read setting is 1, to allow internal nodes to be read
        t = Tree(tree_path + f"run_{iteration}_ancestors.nwk", ete_read_setting)
        names, _ = sample_ancestor_nodes(rejection_threshold, t, iteration)
        sampled_names += names

    sampled_ancestors = pd.read_pickle(ancestor_pkl_path)
    sampled_ancestors = sampled_ancestors.loc[
        sampled_ancestors["id"].isin(sampled_names)
    ]

    return sampled_ancestors
