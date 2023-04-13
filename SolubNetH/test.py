# -*- coding: utf-8 -*-
# Author QFIUNE
# coding=utf-8
# @Time: 2022/6/22 16:01
# @File: test_result.py
# @Software: PyCharm
# @contact: 1760812842@qq.com

import os
import torch as th
import pandas as pd
from mtMolDes import model, Utility
import os
import numpy as np
from mtMolDes.Evaluation import MAE, MSE, RMSE, Spearman
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')
separator = "-" * 90


def get_matrics(y_pred, y_true):
    mae = MAE(y_pred, y_true).detach().numpy()
    mse = MSE(y_pred, y_true).detach().numpy()
    rmse = RMSE(y_pred, y_true).detach().numpy()
    r2 = r2_score(y_pred.detach().numpy(),  y_true.detach().numpy())
    cc = Spearman(y_pred.detach().numpy(),  y_true.detach().numpy())

    print(separator)
    print("%15s %15s %15s %15s %15s" % ("TestMAE", "TestMSE", "TestRMSE", "TestR2", "TestCC"))
    print("%15.3f %15.3f %15.3f %15.3f %15.3f" % (mae, mse, rmse, r2, cc))
    print(separator)
    return round(r2, 3)


def get_picture(y_true, y_pred, _name, R2):
    plt.figure(figsize=(5, 5), dpi=600)
    plt.scatter(y_true, y_pred, marker='o', s=20, label=_name)
    parameter = np.polyfit(y_true,y_pred, 1)
    y = parameter[0] * np.array(y_pred) + parameter[1]
    plt.plot(y_pred, y, color="#130c0e", linewidth=1, label="R2 = " + str(R2))

    plt.xlim(min(min(y_pred), min(y_true)), max(max(y_pred), max(y_true)))
    plt.ylim(min(min(y_pred), min(y_true)), max(max(y_pred), max(y_true)))

    plt.legend(loc='upper left')
        
    plt.xlabel('Exptl. logS')
    plt.ylabel('Pred. logS')

    plt.savefig('test_picture/' + _name + '.jpg')


# -------------------------------------------------------------------------------------------
if __name__ == '__main__':

    num_features = 4
    num_labels = 1
    feature_str = 'h'

    InputDir = "./extended_dataset/"
    for Root, DirNames, FileNames in os.walk(InputDir):
        for idx, FileName in  enumerate(FileNames):
            Name = FileName.split(".")[0]
            FilePath = os.path.join(Root, FileName)

            device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
            data = Utility.LoadGaoData(FilePath, num_features, feature_str, device)

            solubNet = model.GCNNet(num_features, num_labels, feature_str)

            project_path = os.getcwd()
            model_path = project_path + '/models/solubNet6.pt'
            solubNet.load_state_dict(th.load(model_path, map_location='cpu'))


            print("load success")
            print('-'*50)

            y_true = []
            y_pred = []

            res = []
            for i, gx in enumerate(data):
                sml = gx[0]
                true_prop = round(gx[2], 4)
                pred_prop = round(th.sum(solubNet(gx[1]), dim=0).item(), 3)
                print("%5d %15.3f %15.3f" % (i, true_prop, pred_prop))

                y_true.append(true_prop)
                y_pred.append(pred_prop)
                res.append([sml, true_prop, pred_prop])

            R2 = get_matrics(th.from_numpy(np.array(y_pred)), th.from_numpy(np.array(y_true)))
            
            get_picture(y_pred, y_true, Name, R2)

            name = ["smiles", "True", "Prediction"]
            test = pd.DataFrame(columns=name, data=res)
            test.to_csv('test_result/' + Name + '.csv', encoding='utf-8')




