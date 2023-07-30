# Author QFIUNE
# coding=utf-8
# @Time: 2022/8/30 21:56
# @File: train.py
# @Software: PyCharm
# @contact: 1760812842@qq.com
import copy
import csv
import os
import random
import time
import scipy.stats
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import scale
from model import LinearNet
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
root_path = os.path.abspath(os.path.dirname(os.getcwd()))
print('The path to this project is: ', root_path)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def get_morgan_fingerprint(mol):
    fingerprint = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)]
    return fingerprint


def get_Morgan(DataFile):
    csv_reader = csv.reader(open(DataFile))
    next(csv_reader)
    ECFP = []
    labels = []

    for line in csv_reader:
        sml = line[0]
        label = float(line[1])
        labels.append([label])

        mol = Chem.MolFromSmiles(sml)
        morgan = get_morgan_fingerprint(mol)
        # pubchem = get_pubchem_fingerprint(sml)
        ECFP.append(morgan)
    data = np.c_[np.array(ECFP), np.array(labels)]
    print(data.shape)
    return data


class MyDataset(Dataset):
    def __init__(self, feature, label):
        super().__init__()

        self.feature = feature
        self.label = label

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return len(self.feature)


def getDataLoader(feature_label):
    random.seed(42)
    random.shuffle(feature_label)

    feature = feature_label[:, :-1]
    feature = scale(feature)

    label = feature_label[:, -1]
    x_train, y_train = feature[:round(len(feature)*0.9)], label[:round(len(feature)*0.9)]
    x_test, y_test = feature[round(len(feature)*0.9):], label[round(len(feature)*0.9):]
    my_dataset = MyDataset
    train_data = my_dataset(x_train, y_train)
    test_data = my_dataset(x_test, y_test)
    print('train_data: %d   test_data: %d' %(len(train_data), len(test_data)))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    return train_loader, test_loader


def Train(Net, train_loader, criterion, optimizer, scheduler, device):
    train_epoch_loss = []
    train_epoch_pred = []
    train_epoch_true = []
    for step, (feature, label) in enumerate(train_loader):
        feature = torch.tensor(feature).float()
        y_pred = Net(feature).to(device)
        y_true = torch.tensor(label).float()
        loss = criterion(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_epoch_loss.append(loss.item())
        train_epoch_pred += [i.item() for i in y_pred]
        train_epoch_true += [i.item() for i in y_true]

    y_test, y_predict = np.array(train_epoch_true), np.array(train_epoch_pred)
    train_loss = np.average(train_epoch_loss)

    train_mae = mean_absolute_error(y_test, y_predict)
    train_rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    train_r2 = r2_score(y_test, y_predict)
    train_cc = scipy.stats.spearmanr(y_test, y_predict)[0]

    scheduler.step(train_loss)
    return [train_loss, train_mae, train_rmse, train_r2, train_cc]


def Eval(Net, val_loader, device):
    with th.no_grad():
        val_epoch_pred = []
        val_epoch_true = []
        for step, (feature, label) in enumerate(val_loader):
            feature = torch.tensor(feature).float()
            y_pred = Net(feature).to(device)
            y_true = torch.tensor(label).float()

            val_epoch_pred += [i.item() for i in y_pred]
            val_epoch_true += [i.item() for i in y_true]

        y_true, y_pred = np.array(val_epoch_true), np.array(val_epoch_pred)
        # print(y_true[:10], y_pred[:10])
        val_mae = mean_absolute_error(y_true, y_pred)
        val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        val_r2 = r2_score(y_true, y_pred)
        val_cc = scipy.stats.spearmanr(y_true, y_pred)[0]
        return val_mae, val_rmse, val_r2, val_cc


def main(train_loader, val_loader, device):
    epochs = 500
    feature_dim = 1024
    Net = LinearNet(n_feature=feature_dim, n_output=1).to(device)
    optimizer = th.optim.Adam(Net.parameters(), lr=0.001, weight_decay=1.e-4)
    criterion = th.nn.SmoothL1Loss(reduction='mean')

    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10,
                                                        threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                        min_lr=0.000001, eps=1e-08, verbose=False)

    separator = "-" * 200
    print(separator)
    print("%10s %15s %15s %15s %15s %15s %15s %15s %15s %15s" %
          ("Epoch", "TrainLoss", "TrainMAE", "TrainRMSE", "TrainR2", "TrainCC", "ValMAE", "ValRMSE", "ValR2", "ValCC"))
    print(separator)

    MinValMae = 10
    BestModel = None

    for epoch in range(epochs + 1):
        TrainRes = Train(Net, train_loader, criterion, optimizer, scheduler, device)
        ValRes = Eval(Net, val_loader, device)

        print("%10d %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f" %
              (epoch, TrainRes[0], TrainRes[1], TrainRes[2], TrainRes[3], TrainRes[4], ValRes[0], ValRes[1], ValRes[2], ValRes[3]))

        if ValRes[0] <= MinValMae:
            MinValMae = ValRes[0]
            BestModel = copy.deepcopy(Net)
            BestEpoch = epoch
            BestTrainMAE, BestTrainRMSE, BestTrainR2, BestTrainCC = TrainRes[1], TrainRes[2], TrainRes[3], TrainRes[4]
            BestValMAE, BestValRMSE, BestValR2, BestValCC = ValRes[0], ValRes[1], ValRes[2], ValRes[3]

    print(separator)
    print("The best indicator statistics")
    print("%15s %15s %15s %15s %15s %15s %15s %15s %15s" %
          ("BestEpoch", "TrainMAE", "TrainRMSE", "TrainR2", "TrainCC", "ValMAE", "ValRMSE", "ValR2", "ValCC"))
    print("%10d %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f %15.3f" %
          (BestEpoch, BestTrainMAE, BestTrainRMSE, BestTrainR2, BestTrainCC, BestValMAE, BestValRMSE, BestValR2, BestValCC))

    model = BestModel
    th.save(model.state_dict(), 'solubNet.pt')
    return model


if __name__ == '__main__':
    InputFile = "./dataset/ESOL.csv"
    data = get_Morgan(InputFile)
    train_loader, test_loader = getDataLoader(data)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    net = main(train_loader, test_loader, device)

    # separator = "-" * 200
    # epochs = 500
    # feature_dim = 1024
    #
    # print(separator)
    # print('This part is test result!')
    # print(separator)
    #
    # net.eval()
    # Eval(val_loader, device)

    # print(separator)
    # print("%15s %15s %15s %15s %15s" % ("TestACC", "TestPrecision", "TestRecall", "TestF1", "TestAUC"))
    # print("%15.7f %15.7f %15.7f %15.7f %15.7f" % (acc, pre, recall, f1, auc))
    # print(separator)


