import torch as th
import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl import DGLGraph
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import time, os, csv
from sklearn.metrics import r2_score
from .Evaluation import MAE, MSE, RMSE, Spearman
import numpy as np
import pandas as pd
import random
import math

import networkx as nx
import matplotlib.pyplot as plt
from sklearn import preprocessing
# from early_stopping import EarlyStopping
import copy
from torch.optim.lr_scheduler import LambdaLR

import warnings
warnings.filterwarnings('ignore')


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom, stereo, features, tpsa, crippenlogPs, crippenMRs, LaASA, explicit_H=False):
    """
    Method that computes atom level features from rdkit atom object
    :param atom:
    :param stereo:
    :param features:
    :param explicit_H:
    :return: the node features of an atom
    """
    atom_features = [tpsa, crippenlogPs, crippenMRs, LaASA]
    # atom_features += [float(atom.GetProp('_GasteigerCharge'))]  # 获取原子的Gasteiger charge
    # atom_features += [atom.GetDegree()]
    # atom_features += [len(atom.GetNeighbors())]   # 返回该原子的所有邻居原子，以元祖的形式返回
    # atom_features += [atom.GetIsAromatic()]


    # possible_atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'DU']
    # atom_features += one_of_k_encoding_unk(atom.GetSymbol(), possible_atoms)
    # atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    # atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    # atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    # atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    # atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
    #     Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    #     Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D])
    #
    # # ===============================================================================
    # atom_features += [atom.GetDegree()]
    # atom_features += [atom.GetTotalNumHs()]       # 与该原子连接的氢原子个数
    # atom_features += [len(atom.GetNeighbors())]   # 返回该原子的所有邻居原子，以元祖的形式返回
    # atom_features += [atom.GetIsAromatic()]

    # atom_features += [atom.GetExplicitValence()]
    # atom_features += [atom.GetFormalCharge()]
    # atom_features += [atom.GetIsAromatic()]
    # atom_features += [atom.GetHybridization()]
    # # ===============================================================================
    #
    # atom_features += [int(i) for i in list("{0:06b}".format(features))]
    #
    # if not explicit_H:
    #     atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    #
    # try:
    #     atom_features += one_of_k_encoding_unk(stereo, ['R', 'S'])
    #     atom_features += [atom.HasProp('_ChiralityPossible')]
    # except Exception as e:
    #
    #     atom_features += [False, False] + [atom.HasProp('_ChiralityPossible')]
    # print(atom_features)
    return np.array(atom_features)


def get_bond_features(bond):
    """
    Method that computes bond level features from rdkit bond object
    :param bond: rdkit bond object
    :return: bond features, 1d numpy array
    """

    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats)


def ParseSMILES(sml, num_features, feature_str, device):
    """Transform a SMILES code to RDKIT molecule and DGL graph.

    Args:
        sml (str):          The SMILES code.
        num_features (int): The dimension of features for all atoms.
        feature_str (str):  The string to access the node features.
        device (str):       The device (CPU or GPU) to store the DGL graph.

    Returns:
        (mol, graph): The RDKIT molecule and DGL graph.
    """

    mol = Chem.MolFromSmiles(sml)
    if mol is None:
        raise ValueError("Invalid SMILES code: %s" % (sml))

    features = rdDesc.GetFeatureInvariants(mol)
    # AllChem.ComputeGasteigerCharges(mol)
    # TPSA = rdMolDescriptors._CalcTPSAContribs(mol)

    # Calculation of the properties.
    AllChem.ComputeGasteigerCharges(mol)
    (CrippenlogPs, CrippenMRs) = zip(*(Chem.rdMolDescriptors._CalcCrippenContribs(mol)))
    TPSAs = Chem.rdMolDescriptors._CalcTPSAContribs(mol)
    (LaASAs, x) = Chem.rdMolDescriptors._CalcLabuteASAContribs(mol)


    graph = dgl.DGLGraph()

    stereo = Chem.FindMolChiralCenters(mol)
    chiral_centers = [0] * mol.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]

    graph.add_nodes(mol.GetNumAtoms())  # 添加节点
    node_features = []
    edge_features = []
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)

        atom_i_features = get_atom_features(atom_i, chiral_centers[i], features[i], TPSAs[i],
                                            CrippenlogPs[i], CrippenMRs[i], LaASAs[i])
        node_features.append(atom_i_features)

        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                graph.add_edges(i, j)  # 添加边
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    graph.ndata['h'] = F.normalize(th.from_numpy(np.array(node_features)).to(th.float32))
    graph.edata['w'] = th.from_numpy(np.array(edge_features))
    # print(graph)
    return graph


def LoadGaoData(fn, num_features, feature_str, device):
    """Load data contributed by Dr. Peng Gao.

    Args:
        fn (str):           The file name.
        num_features (int): The dimension of features for all atoms.
        feature_str (str):  The string to access the node features.
        device (str):       The device (CPU or GPU) to store the DGL graph.

    Returns:
        [(graph, property)]: The DGL graph and property.
    """
    print("Load GaoDataSet from %s ... " % (fn), flush=True, end="")
    t0 = time.time()
    csv_reader = csv.reader(open(fn))
    next(csv_reader)
    data = []
    for line in csv_reader:
        graph = ParseSMILES(line[0], num_features, feature_str, device)
        prop = float(line[1])
        data.append([line[0], graph, prop])

    t1 = time.time()
    dur = t1 - t0
    print("done (%d lines, %.3f seconds) " % (len(data) + 1, dur), flush=True)
    return data

