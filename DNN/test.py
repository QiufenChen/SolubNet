# Author QFIUNE
# coding=utf-8
# @Time: 2022/9/11 13:23
# @File: test.py
# @Software: PyCharm
# @contact: 1760812842@qq.com

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import scale

from model import LinearNet
import torch as th
from Evaluation import ACC, Precision, Recall, F1_Score, AUC
import numpy as np
import warnings

warnings.filterwarnings("ignore")
root_path = os.path.abspath(os.path.dirname(os.getcwd()))
print('The path to this project is: ', root_path)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from LRPExplanation import LRPModel


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
    feature = feature_label[:, :-1]
    feature = scale(feature)

    label = feature_label[:, -1]
    x_train, y_train = feature[:round(len(feature)*0.9)], label[:round(len(feature)*0.9)]
    x_test, y_test = feature[round(len(feature)*0.9):], label[round(len(feature)*0.9):]

    my_dataset = MyDataset
    train_data = my_dataset(x_train, y_train)
    test_data = my_dataset(x_test, y_test)

    print('train_data: %d   test_data: %d' %(len(train_data), len(test_data)))

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader


# ----------------------------------------------------------------------------------------------------------------------
topo = np.load('../dataset/data1/feature/Topo.npy')
label = np.load('../dataset/data1/feature/label.npy')

feature_label = np.c_[topo, label]
print(feature_label.shape)

separator = "-" * 80
epochs = 100
feature_dim = 79
weights = [1, 7.0]
save_name = 'XXX'

train_loader, test_loader = getDataLoader(feature_label)
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
net = LinearNet(n_feature=feature_dim, n_output=2).to(device)


model_path = './model/morgan_avalon_maccs_1-1.pt'
net.load_state_dict(th.load(model_path, map_location='cpu'))
train_loader, test_loader = getDataLoader(feature_label)

print(separator)
print('This part is test test-result!')
print(separator)

net.eval()
test_acc, test_prec, test_recall, test_f1, test_auc = 0, 0, 0, 0, 0, 0
pred = []
test = []

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

with torch.no_grad():
    for step, (feature, label) in enumerate(test_loader):
        # print(feature,label)
        feature = torch.tensor([item.cpu().detach().numpy() for item in feature]).float()
        y_pred = net(feature).to(device)
        y_true = label.long()
        y_pred = th.argmax(y_pred, axis=1)
        pred.append(y_pred.numpy())
        test.append(y_true.numpy())

       
# cm = confusion_matrix(test, pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['non-activate', 'activate'])
# disp.plot(cmap='Blues')
# plt.show()
# # plt.savefig('test-result/' + 'ConfusionMatrix-' + save_name + '.png')
#
# acc = ACC(test, pred)
# pre = Precision(test, pred)
# recall = Recall(test, pred)
# f1 = F1_Score(test, pred)
# auc = AUC(test, pred)
#
# print(separator)
# print("%15s %15s %15s %15s %15s" % ("TestACC", "TestPrecision", "TestRecall", "TestF1", "TestAUC"))
# print("%15.7f %15.7f %15.7f %15.7f %15.7f" % (acc, pre, recall, f1, auc))
# print(separator)

