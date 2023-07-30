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
from LRPExplanation import LRPModel
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
import os
import numpy as np
# from mtMolDes.Evaluation import MAE, MSE, RMSE, PCC
from mtMolDes.Evaluation import MAE, MSE, RMSE, Spearman
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')


separator = "-" * 90


def get_atom_labels(mol):
    atomLabels = dict()
    for atom in mol.GetAtoms():
        atomLabels[atom.GetIdx()] = atom.GetSymbol() + str(atom.GetIdx())
    return atomLabels


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def visualization(graph, R, sml_id, sml, y_true, y_pred):
    mol = Chem.MolFromSmiles(sml)

    plt.figure(figsize=(15, 13), dpi=600)

    text = "True property: " + str(y_true) + '        ' + 'Predicted property: ' + str(y_pred)
    g = graph.to_networkx(node_attrs='h', edge_attrs='')  # 转换 dgl graph to networks
    pos = nx.kamada_kawai_layout(g)

    pos_higher = {}
    for k, v in pos.items():
        if (v[1] > 0):
            pos_higher[k] = (v[0] - 0.04, v[1] + 0.04)
        else:
            pos_higher[k] = (v[0] - 0.04, v[1] - 0.04)

    node_labels = get_atom_labels(mol)

    plt.subplot(1, 2, 1)
    nx.draw(g, with_labels=True, pos=pos, node_size=800, labels=node_labels, font_color="black")
    plt.text(0, 0, text, fontsize=10, transform=plt.gca().transAxes)
    plt.title("Molecular graph without explanation")
    # ax[0][0].set_xlabel("True property: " + str(y_true), fontsize=15, fontweight='bold')

    # 修改坐标轴字体及大小
    # plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')  # 设置大小及加粗
    # plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')

    nodes = []
    weights = []
    save_weight = []
    for idx_atom in range(graph.nodes().shape[0]):
        weight = float(R[idx_atom])
        weights.append(weight)
        nodes.append((idx_atom, {"weight": weight}))

        save_weight.append([idx_atom, weight])
    g.add_nodes_from(nodes)

    # save weight
    name0 = ['atom_num', 'mean_weight']
    test0 = pd.DataFrame(columns=name0, data=save_weight)
    test0.to_csv('./weight/' + sml_id + '.csv', encoding='utf-8')

    # 解释之后的图
    plt.subplot(1, 2, 2)
    cmap = plt.cm.get_cmap('Greens')
    nx.draw(g, with_labels=True, pos=pos, node_color=weights, labels=node_labels, cmap=cmap, node_size=800, font_color="black")
    plt.title("Molecular graph after explanation")
    # ax[0][1].set_xlabel('Predicted property: ' + str(y_pred), fontsize=15, fontweight='bold')
    plt.text(0, 0, text, fontsize=10, transform=plt.gca().transAxes)
    # plt.show()
    plt.savefig('./graphs/' + sml_id + ".png", format="PNG", dpi=600)

    # 用RDKit可视化分子
    mol = Chem.MolFromSmiles(sml)
    opts = DrawingOptions()
    opts.includeAtomNumbers = True
    opts.bondLineWidth = 2.8

    # plt.subplot(2, 2, 3)
    draw = Draw.MolToImage(mol, options=opts, size=(600, 600))
    draw.save('./molecules_atoms/' + sml_id + '.png')

    # plt.subplot(2, 2, 4)
    Draw.MolToImage(mol, size=(600, 600), kekulize=True)
    Draw.ShowMol(mol, size=(600, 600), kekulize=False)
    Draw.MolToFile(mol, './molecules_num_atoms/' + sml_id + '.png', size=(600, 600), kekulize=False)


def get_matrics(y_pred, y_true):
    # mae = MAE(th.from_numpy(np.array(y_pred)), th.from_numpy(np.array(y_true)))
    # mse = MSE(th.from_numpy(np.array(y_pred)), th.from_numpy(np.array(y_true)))
    # rmse = RMSE(th.from_numpy(np.array(y_pred)), th.from_numpy(np.array(y_true)))
    mae = np.round(MAE(y_pred, y_true).detach().numpy(),3)
    mse = np.round(MSE(y_pred, y_true).detach().numpy(),3)
    rmse = np.round(RMSE(y_pred, y_true).detach().numpy(),3)
    r2 = np.round(r2_score(y_pred.detach().numpy(),  y_true.detach().numpy()), 3)
    cc = np.round(Spearman(y_pred.detach().numpy(),  y_true.detach().numpy()),3)


    # print(np.round(mae.numpy(), 4), np.round(mse.numpy(), 4), np.round(rmse.numpy(), 4))
    print(separator)
    print("%15s %15s %15s %15s %15s" %
          ("TestMAE", "TestMSE", "TestRMSE", "TestR2", "TestCC"))
    print("%15.7f %15.7f %15.7f %15.7f %15.7f" % (mae, mse, rmse, r2, cc))
    print(separator)
    return r2


def get_picture(y_true, y_pred, _name, _color, R2):
    # print(len(y_true), len(y_pred))
    # print('y_true:',y_true)
    # print('y_pred:',y_pred)
    plt.figure(figsize=(5, 5), dpi=600)
    plt.scatter(y_true, y_pred, marker='o', s=20, label=_name+"_Random")
    parameter = np.polyfit(y_true,y_pred, 1)
    y = parameter[0] * np.array(y_pred) + parameter[1]
    plt.plot(y_pred, y, color="#130c0e", linewidth=1, label="R2 = " + str(R2))

    plt.xlim(min(min(y_pred), min(y_true)), max(max(y_pred), max(y_true)))
    plt.ylim(min(min(y_pred), min(y_true)), max(max(y_pred), max(y_true)))

    plt.legend(loc='upper left')
    
    # plt.text(min(min(y_pred), min(y_true))+2, max(max(y_pred), max(y_true))-0.5,"R2 = " + str(R2), ha='center', va='center')
        
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')

    plt.savefig('test_picture/' + _name + '.jpg')


# -------------------------------------------------------------------------------------------
if __name__ == '__main__':

    num_features = 4
    num_labels = 1
    feature_str = 'h'

    # data_fn_1 = "./extended_dataset/20-drugs.csv"
    # data_fn_2 = "./extended_dataset/CuiNovel_62.csv"
    # data_fn_3 = "./extended_dataset/Boobier_100.csv"
    # data_fn_4 = "./extended_dataset/llinas2020_132.csv"
    # data_fn_5 = "./extended_dataset/llinas2020_set1_100.csv"
    # data_fn_6 = "./extended_dataset/llinas2020_set2_32.csv"
    # data_fn_7 = "./extended_dataset/ESOL_1128.csv"
    # FileLi = [data_fn_1, data_fn_2, data_fn_3, data_fn_4, data_fn_5, data_fn_6, data_fn_7]
    # NameLi = ["Gao20", "CuiNovel62", "Boobier100", "LlinasAll", "LlinasSet1", "LlinasSet2", "Deneley"]

    InputDir = "./extended_dataset/"
    for Root, DirNames, FileNames in os.walk(InputDir):
        for idx, FileName in  enumerate(FileNames):
            Name = FileName.split(".")[0]
            FilePath = os.path.join(Root, FileName)

            device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
            data = Utility.LoadGaoData(FilePath, num_features, feature_str, device)

            solubNet = model.GCNNet(num_features, num_labels, feature_str)

            project_path = os.getcwd()
            model_path = project_path + '/models/solubNet.pt'
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

                # Start explaining molecules_atoms
                # sml = data[i][0]
                # graph = data[i][1]
                # R = LRPModel(solubNet)(data[i][1])
                # for idx, r in enumerate(R):
                #     # print(r.shape, th.sum(r))
                #     print(idx, th.sum(r), r)

                # visualization(graph, R[-1], sml_id, sml, true_prop, pred_prop)


            R2 = get_matrics(th.from_numpy(np.array(y_pred)), th.from_numpy(np.array(y_true)))
            ColorLi = ['#007d65', '#007d65', '#007d65', '#007d65', '#007d65', '#007d65', '#007d65', '#007d65']
           
            print(len(y_pred), len(y_true))
            get_picture(y_pred, y_true, Name, ColorLi[idx], R2)

            name = ["smiles", "True", "Prediction"]
            test = pd.DataFrame(columns=name, data=res)
            test.to_csv('test_result/' + Name + '.csv', encoding='utf-8')


