# Author QFIUNE
# coding=utf-8
# @Time: 2023/3/23 10:33
# @File: DataSplit.py
# @Software: PyCharm
# @contact: 1760812842@qq.com

"""
Reference to: https://github.com/XinhaoLi74/molds/blob/e3ce4504a6d6f5c22f7f0155598496675001f0bc/molds/DSsplitter.py
"""

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union
import random
import pandas as pd

import os


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("This path is exit!")


def generate_scaffold(smi: str, include_chirality: bool = False):
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.
    - parameters:
        - mol: A SMILES string.
        - include_chirality: Whether to include chirality.
    - return: the SMILES of the scaffold
    """
    mol = Chem.MolFromSmiles(smi)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_to_smiles(data, use_indices: bool = False):
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.
    - parameters:
        - mols: A list of SMILES strings.
        - use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    - return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, smi in enumerate(data):
        scaffold = generate_scaffold(smi)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(smi)
    return scaffolds


def scaffold_split(data,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = True,
                   seed: int = 0):
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.
    - parameters:
        - data: A list of smiles.
        - sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
        - balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
        - seed: Seed for reproducitility.
    - return: two tuples
        - (1) containing the train, validation, and test splits of the data (SMILES)
        - (2) containing the train, validation, and test splits of the index.
    """
    assert sum(sizes) == 1
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train_index, val_index, test_index = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data, use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets, in this case, the largest set will always be put into training set.
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train_index) + len(index_set) <= train_size:
            train_index += index_set
            train_scaffold_count += 1
        elif len(val_index) + len(index_set) <= val_size:
            val_index += index_set
            val_scaffold_count += 1
        else:
            test_index += index_set
            test_scaffold_count += 1

    print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                     f'train scaffolds = {train_scaffold_count:,} | '
                     f'val scaffolds = {val_scaffold_count:,} | '
                     f'test scaffolds = {test_scaffold_count:,}')

    # Map from indices to data
    train = [data[i] for i in train_index]
    val = [data[i] for i in val_index]
    test = [data[i] for i in test_index]

    return train_index, val_index, test_index


InputFile = "dataset/Delaney1144.csv"
SavePath = "dataset/Delaney/"

df = pd.read_csv(InputFile, header=0, sep=',', dtype=str)
smiles = df["smiles"].tolist()
labels = df["LogS"].tolist()

train_inds, valid_inds, test_inds = scaffold_split(smiles, sizes=(0.8, 0.1, 0.1))
name = ["smiles", "LogS"]
TrainData = [[smiles[idx], float(labels[idx])] for idx in train_inds]
ValData = [[smiles[idx], float(labels[idx])] for idx in valid_inds]
TestData = [[smiles[idx], float(labels[idx])] for idx in test_inds]

TrainData_df = pd.DataFrame(TrainData, columns=name)
ValidData_df = pd.DataFrame(ValData, columns=name)
TestData_df = pd.DataFrame(TestData, columns=name)

mkdir(SavePath)
TrainData_df.to_csv(SavePath + "DelaneyTrain.csv", index=False)
ValidData_df.to_csv(SavePath + "DelaneyValid.csv", index=False)
TestData_df.to_csv(SavePath + "DelaneyTest.csv", index=False)
