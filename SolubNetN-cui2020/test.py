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
from mtMolDes.Evaluation import MAE, MSE, RMSE, PCC

import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')


separator = "-" * 20


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
    mae = MAE(y_pred, y_true)
    mse = MSE(y_pred, y_true)
    rmse = RMSE(y_pred, y_true)
    pcc = PCC(y_pred, y_true)

    # print(np.round(mae.numpy(), 4), np.round(mse.numpy(), 4), np.round(rmse.numpy(), 4))
    print(separator)
    print("%15s %15s %15s %15s" % ("Test-MAE", "Test-MSE", "Test-RMSE", "Test-PCC"))
    print("%15.4f %15.4f %15.4f %15.4f" % (mae, mse, rmse, pcc))
    print(separator)



def get_picture(y_true, y_pred, _name, _color):
    plt.figure(figsize=(5, 5), dpi=600)
    plt.scatter(y_true, y_pred, c=_color)
    parameter = np.polyfit(y_pred, y_true, 1)
    y = parameter[0] * np.array(y_pred) + parameter[1]
    plt.plot(y_pred, y, color='#FF8C00', linewidth=3)

    plt.xlim(min(min(y_pred), min(y_true)), max(max(y_pred), max(y_true)))
    plt.ylim(min(min(y_pred), min(y_true)), max(max(y_pred), max(y_true)))

    plt.legend(loc='upper right', labels=['Fitted Value', _name])
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')

    plt.savefig('test_picture/' + _name + '.jpg')


# -------------------------------------------------------------------------------------------
if __name__ == '__main__':

    num_features = 1
    num_labels = 1
    feature_str = 'h'
    EPSILON = 0.01

    data_fn = 'extended-data/' + '20-drugs' + '.csv'
    # data_fn = 'extended-data/' + 'intrinsic' + '.csv'
    # data_fn = 'extended-data/' + 'llinas2020_raw' + '.csv'

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    data = Utility.LoadGaoData(data_fn, num_features, feature_str, device)

    solubNet = model.GCNNet(num_features, num_labels, feature_str)

    project_path = os.getcwd()
    model_path = project_path + '/solubNet.pt'
    solubNet.load_state_dict(th.load(model_path, map_location='cpu'))


    print("load success")
    print('-'*50)

    y_true = []
    y_pred = []
    idx = []
    res = []
    for i, gx in enumerate(data):
        sml_id = 'molecule_' + str(i+1)
        true_prop = round(gx[2], 4)
        pred_prop = round(th.sum(solubNet(gx[1]), dim=0).item(), 4)
        print("%5d %15.4f %15.4f" % (i, true_prop, pred_prop))

        # idx.append(i)
        y_true.append(true_prop)
        y_pred.append(pred_prop)
        res.append([true_prop, pred_prop])

        # Start explaining molecules_atoms
        sml = data[i][0]
        graph = data[i][1]
        # R = LRPModel(solubNet)(data[i][1])
        # for idx, r in enumerate(R):
        #     # print(r.shape, th.sum(r))
        #     print(idx, th.sum(r), r)

        # visualization(graph, R[-1], sml_id, sml, true_prop, pred_prop)


    get_matrics(th.from_numpy(np.array(y_pred)), th.from_numpy(np.array(y_true)))
    # color = ['#8A2BE2', '#008B8B', '#228B22', '#008080', '#3cb371', '#ffa500', '#6a5acd']
    get_picture(y_pred, y_true, 'Solub', '#228B22')

    name = ['true_prop', 'pred_prop']
    test = pd.DataFrame(columns=name, index=idx, data=res)
    test.to_csv('test_result/intrinsic.csv', encoding='utf-8')


