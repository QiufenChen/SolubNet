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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
from torch.optim.lr_scheduler import LambdaLR,StepLR,ExponentialLR,CosineAnnealingLR

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


def criterionR(output, target):
    target_mean = th.mean(target)
    ss_tot = th.sum((target - target_mean) ** 2)
    ss_res = th.sum((target - output) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def Train(net, data_train, data_valid, learning_rate, batch_size, max_epochs, output_freq, device):
    """Train the net. The models will be saved.

    Args:
        net (pytorch module):       The net to train.
        data ([(graph, property)]): The data set.
        learning_rate (float):      The learning rate for optimization.
        batch_size (int):           The batch size.
        max_epochs (int):           The number of epochs to train.
        output_freq (int):          The frequency of output.
        device (str):               The device (CPU or GPU) to store the DGL graph.
    """

    # net.to(device)
    # random.seed(1023)
    # random.shuffle(all_data)
    # ratio = 0.9
    # num = len(all_data)
    # data_train = all_data[: round(num * ratio)]
    # data_Valid = all_data[round(num * ratio):]
    
    print("# of training graphs/labels:      %d" % (len(data_train)))
    print("# of Validing graphs/labels:      %d" % (len(data_valid)))

    # TrainDataset = [[gx[0], gx[2]] for gx in data_train]
    # ValidDataset = [[gx[0], gx[2]] for gx in data_Valid] 
    # name = ['smiles', 'LogS']
    # df_train = pd.DataFrame(columns=name, data=TrainDataset)
    # df_Valid = pd.DataFrame(columns=name, data=ValidDataset)  
    # df_train.to_csv('dataset/train.csv', index=False)
    # df_Valid.to_csv('dataset/Valid.csv', index=False)

    net.to(device)
    random.seed(42)
    random.shuffle(data_train)

    train_graphs = [gx[1] for gx in data_train]
    train_labels = th.tensor([gx[2] for gx in data_train]).to(device)
    Valid_graphs = [gx[1] for gx in data_valid]
    Valid_labels = th.tensor([gx[2] for gx in data_valid]).to(device)

    # TrainDataset = [[gx[0], gx[2]] for gx in data_train]
    # ValidDataset = [[gx[0], gx[2]] for gx in data_Valid]
    # name = ['smiles', 'LogS']
   
    # df_train = pd.DataFrame(columns=name, data=TrainDataset)
    # df_Valid = pd.DataFrame(columns=name, data=ValidDataset)
    
    # df_train.to_csv('dataset/train.csv', index=False)
    # df_Valid.to_csv('dataset/Valid.csv', index=False)

    # =================================== optimizer ================================
    # optimizer = th.optim.RMSprop(net.parameters(), lr=learning_rate, alpha=0.9)
    optimizer = th.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = th.optim.Adamax(net.parameters(), lr=learning_rate, weight_decay=1.e-4)
    # optimizer = th.optim.ASGD(net.parameters(), lr=learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    # ==============================================================================

    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=3, threshold=0.0001,
        threshold_mode='rel', cooldown=0, min_lr=0.000001, eps=1e-08, verbose=False)
        
    # import torchmetrics.functional as tff
    # criterionR = tff.r2_score
    criterionL = th.nn.MSELoss(reduction='mean')     # MSE
    # criterion = th.nn.L1Loss(reduction='mean')      # MAE
    # criterion = th.nn.SmoothL1Loss(reduction='mean')  # SmoothL1Loss

    # A closure to calculate loss.
    def getY(gs, ps):
        num_ps = ps.shape[0]
        p0s = th.zeros(num_ps).to(device)  # The predicted properties.
        # print(p0s.shape, p0s)
        for i in range(num_ps):
            # print(ps)
            # print(net(gs[i]).shape, th.mean(net(gs[i]), dim=0))
            p0s[i] = th.sum(net(gs[i].to(device)), dim=0)

        return p0s, ps  # The predicted and true properties.

    # Set mini-batch.
    batch_idx = None
    train_num = len(data_train)
    if batch_size >= train_num:
        batch_idx = [[0, train_num]]
    else:
        batch_idx = [[i * batch_size, (i + 1) * batch_size] for i in range(train_num // batch_size)]
        if batch_idx[-1][1] != train_num: batch_idx.append([batch_idx[-1][1], train_num])


    # Output.
    print(">>> Training of the Model >>>")
    print("Start at: ", time.asctime(time.localtime(time.time())))
    print("PID:      ", os.getpid())
    print("Learning rate:               %4.E" % (learning_rate))
    print("Batch size:                  %d" % (batch_size))
    print("Maximum epochs:              %d" % (max_epochs))
    print("Output frequency:            %d" % (output_freq))
    print("Device:                      %s" % (device))
    separator = "-" * 200
    print(separator)
    print("%10s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s" %
          ("Epoch", "TrainLoss", "TrainMAE", "TrainRMSE", "TrainR2", "TrainCC",
           "ValidLoss", "ValidMAE", "ValidRMSE", "ValidR2", "ValidCC", "Time(s)"))
    print(separator)

    # Training begins.
    t_begin = time.time()
    t0 = t_begin

    MinValMae = 10
    BestTrainMAE, BestTrainRMSE, BestTrainR2, BestTrainCC = 0,0,0,0
    BestValidMAE, BestValidRMSE, BestValidR2, BestValidCC = 0,0,0,0
    BestEpoch = 0
    BestModel = None

    train_epochs_loss = []
    valid_epochs_loss = []

    net.train()
    for epoch in range(max_epochs + 1):
        # n = len(batch_idx)
        # w_loss, w_mae, w_rmse, w_r2, w_cc = 0, 0, 0, 0, 0
        train_epoch_loss = []
        train_epoch_pred = []
        train_epoch_true = []

        for idx in batch_idx:
            idx0 = idx[0]
            idx1 = idx[1]
            # print(train_graphs[idx0:idx1], train_labels[idx0:idx1])
            y_pred, y_true = getY(train_graphs[idx0:idx1], train_labels[idx0:idx1])

            loss = th.sqrt(criterionL(y_pred, y_true)) - 0.1 * criterionR(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss.append(loss.item())
            y_pred = [i.item() for i in y_pred]
            y_true = [i.item() for i in y_true]
            train_epoch_pred += y_pred
            train_epoch_true += y_true

        y_Valid, y_predict = np.array(train_epoch_true), np.array(train_epoch_pred)

        train_loss = np.average(train_epoch_loss)
        # train_epochs_loss.append(train_loss)

        train_mae = mean_absolute_error(y_Valid, y_predict)
        train_rmse = np.sqrt(mean_squared_error(y_Valid, y_predict))
        train_r2 = r2_score(y_Valid, y_predict)
        train_cc = Spearman(y_Valid, y_predict)

        scheduler.step(train_loss)

        with th.no_grad():
            y_pred, y_true = getY(Valid_graphs, Valid_labels)
            valid_loss = th.sqrt(criterionL(y_pred, y_true)) - 0.1 * criterionR(y_pred, y_true)
            valid_epochs_loss.append(valid_loss.item())
            y_pred = np.array([i.item() for i in y_pred])
            y_true = np.array([i.item() for i in y_true])
            valid_mae = mean_absolute_error(y_true, y_pred)
            valid_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            valid_r2 = r2_score(y_true, y_pred)
            valid_cc = Spearman(y_true, y_pred)

            t1 = time.time()
            dur = t1 - t0
            print("%10d %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f" %
                  (epoch, train_loss, train_mae, train_rmse, train_r2, train_cc, valid_loss, valid_mae, valid_rmse, valid_r2, valid_cc, dur))

            if valid_mae <= MinValMae:
                MinValMae = valid_mae
                BestModel = copy.deepcopy(net)
                BestEpoch = epoch
                BestTrainMAE, BestTrainRMSE, BestTrainR2, BestTrainCC=train_mae, train_rmse, train_r2, train_cc
                BestValidMAE, BestValidRMSE, BestValidR2, BestValidCC=valid_mae, valid_rmse, valid_r2, valid_cc
   
    model = BestModel
    th.save(model.state_dict(), './models/solubNet.pt')

    # =========================plot==========================
    # plt.figure(figsize=(15, 8))
    # plt.plot(train_epochs_loss, '-o', label="train_loss")
    # plt.plot(valid_epochs_loss, '-o', label="valid_loss")
    # plt.title("epochs_loss")
    # plt.legend()
    # plt.show()

    t_end = time.time()
    print(separator)
    print("The best indicator statistics")
    print("%15s %15s %15s %15s %15s %15s %15s %15s %15s" %
          ("BestEpoch", "TrainMAE", "TrainRMSE", "TrainR2", "TrainCC", "ValidMAE", "ValidRMSE", "ValidR2", "ValidCC"))
    print("%10d %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f" %
          (BestEpoch, BestTrainMAE, BestTrainRMSE, BestTrainR2, BestTrainCC, BestValidMAE, BestValidRMSE, BestValidR2, BestValidCC))
    print("Total training time: %.4f seconds" % (t_end - t_begin))
    print(">>> Training of the Model Accomplished! >>>")
    print("End at: ", time.asctime(time.localtime(time.time())))
