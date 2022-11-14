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
from .Evaluation import PCC, MAE, MSE, RMSE
import numpy as np
import random
import math

import networkx as nx
import matplotlib.pyplot as plt
from sklearn import preprocessing


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

    # Parse SMILES to a molecule.
    mol = Chem.MolFromSmiles(sml)
    if mol is None:
        raise ValueError("Invalid SMILES code: %s" % (sml))
    # Build DGL graph from mol.
    edges1 = []
    edges2 = []
    atoms = mol.GetAtoms()
    for atom in atoms:
        for bond in atom.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            edges1.append(idx1)
            edges1.append(idx2)
            edges2.append(idx2)
            edges2.append(idx1)
    graph = dgl.graph((th.tensor(edges1).to(device), th.tensor(edges2).to(device)))
    # Add features.
    num_atoms = len(atoms)
    graph.ndata[feature_str] = th.zeros(num_atoms, num_features).to(device)
    for i, atom in enumerate(atoms):
        graph.ndata[feature_str][i, 0] = atom.GetAtomicNum()
    # Done parsing.
    return (mol, graph)


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
        # print(line[0])
        _, graph = ParseSMILES(line[0], num_features, feature_str, device)
        prop = float(line[1])
        data.append([line[0], graph, prop])

    t1 = time.time()
    dur = t1 - t0
    print("done (%d lines, %.3f seconds) " % (len(data) + 1, dur), flush=True)
    return data


def Train(net, training_data, learning_rate, batch_size, max_epochs, output_freq, save_fn_prefix, device):
    """Train the net. The models will be saved.

    Args:
        net (pytorch module):       The net to train.
        data ([(graph, property)]): The data set.
        training_ratio (float):     The ratio of training data.
        learning_rate (float):      The learning rate for optimization.
        batch_size (int):           The batch size.
        max_epochs (int):           The number of epochs to train.
        output_freq (int):          The frequency of output.
        save_fn_prefix (str):       The net will save as save_fn_prefix+".pkl".
        device (str):               The device (CPU or GPU) to store the DGL graph.
    """

    net.to(device)

    random.seed(1024)
    random.shuffle(training_data)

    num_training_data = len(training_data)

    training_graphs = [gx[1] for gx in training_data]
    training_labels = th.tensor([gx[2] for gx in training_data]).to(device)
   

    # =================================== optimizer ================================
    optimizer = th.optim.RMSprop(net.parameters(), lr=learning_rate, alpha=0.9)
    # optimizer = th.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1.e-4)
    # optimizer = th.optim.Adamax(net.parameters(), lr=learning_rate, weight_decay=1.e-4)
    # optimizer = th.optim.ASGD(net.parameters(), lr=learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    # ==============================================================================

    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=10, threshold=0.0000001,
        threshold_mode='rel', cooldown=0, min_lr=0.000001, eps=1e-08, verbose=False)

    # criterion = th.nn.MSELoss(reduction='mean')     # MSE
    # criterion = th.nn.L1Loss(reduction='mean')      # MAE
    criterion = th.nn.SmoothL1Loss(reduction='mean')  # SmoothL1Loss

    # A closure to calculate loss.
    def getY(gs, ps):
        num_ps = ps.shape[0]
        p0s = th.zeros(num_ps).to(device)  # The predicted properties.
        # print(p0s.shape, p0s)
        for i in range(num_ps):
            # print(ps)
            # print(net(gs[i]).shape, th.mean(net(gs[i]), dim=0))
            p0s[i] = th.sum(net(gs[i]), dim=0)

        return p0s, ps  # The predicted and true properties.

    # Set mini-batch.
    batch_idx = None
    if batch_size >= num_training_data:
        batch_idx = [[0, num_training_data]]
    else:
        batch_idx = [[i * batch_size, (i + 1) * batch_size] for i in range(num_training_data // batch_size)]
        if batch_idx[-1][1] != num_training_data: batch_idx.append([batch_idx[-1][1], num_training_data])

    # Output.
    print(">>> Training of the Model >>>")
    print("Start at: ", time.asctime(time.localtime(time.time())))
    print("PID:      ", os.getpid())
    print("# of training graphs/labels: %d" % (num_training_data))
    print("Learning rate:               %4.E" % (learning_rate))
    print("Batch size:                  %d" % (batch_size))
    print("Maximum epochs:              %d" % (max_epochs))
    print("Output frequency:            %d" % (output_freq))
    print("Params filename prefix:      %s" % (save_fn_prefix))
    print("Device:                      %s" % (device))
    separator = "-" * 150
    print(separator)
    print("%10s %15s %15s %15s %15s %s" %
          ("Epoch", "TrainingLoss", "TrainMAE", "TrainRMSE", "Time(s)", "SavePrefix"))
    print(separator)

    # Training begins.
    t_begin = time.time()
    t0 = t_begin

    net.train()
    w_loss, w_mae, w_mse, w_rmse = 0, 0, 0, 0
    for epoch in range(max_epochs + 1):
        # Do mini-batch.
        n = len(batch_idx)
        for idx in batch_idx:
            idx0 = idx[0]
            idx1 = idx[1]

            y_pred, y_ture = getY(training_graphs[idx0:idx1], training_labels[idx0:idx1])

            # print(th.any(th.isnan(y_pred)),th.any(th.isnan(y_ture)))

            # Calculate loss and other evaluation indicators.
            train_loss = criterion(y_pred, y_ture)
            mae = MAE(y_pred, y_ture)
            mse = MSE(y_pred, y_ture)
            rmse = RMSE(y_pred, y_ture)

            # Move forward.
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            train_loss.backward()
            optimizer.step()

            w_loss += train_loss.detach().item()
            w_mae += mae.detach().item()
            w_mse += mse.detach().item()
            w_rmse += rmse

        w_loss /= n
        w_mae /= n
        w_mse /= n
        w_rmse /= n

        scheduler.step(w_loss)

        prefix = save_fn_prefix + "-" + str(epoch)
        t1 = time.time()
        dur = t1 - t0
        t0 = t1
       
        if epoch / 10 == 0:
            print("%10d %15.7f %15.7f %15.7f %15.7f %s" % (epoch, w_loss, w_mae, w_rmse, dur, prefix))
            
            
    th.save(net.state_dict(), 'solubNet.pt')
    t_end = time.time()
    print(separator)

    print("Final loss: %.4f, Final mae: %.4f, Final mse: %.4f, Final rmse: %.4f" % (w_loss, w_mae, w_mse, w_rmse))
    print("Total training time: %.4f seconds" % (t_end - t_begin))
    print(">>> Training of the Model Accomplished! >>>")
    print("End at: ", time.asctime(time.localtime(time.time())))