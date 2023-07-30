import matplotlib
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

import os
import torch as th
import pandas as pd
from mtMolDes import model, Utility
from LRPExplanation import LRPModel
import networkx as nx
from rdkit import Chem
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

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


def visualization(graph, R, id, sml):
    g = graph.to_networkx(node_attrs='h', edge_attrs='') 
    nodes = []
    atom_weights = []
    save_weight = []
    for idx_atom in range(graph.nodes().shape[0]):
        weight = float(sum(R[idx_atom]))
        atom_weights.append(weight)
        nodes.append((idx_atom, {"weight": weight}))

        save_weight.append([idx_atom, weight])
    g.add_nodes_from(nodes)

    atom_weights = np.array(atom_weights)
    min_value = min(atom_weights)
    max_value = max(atom_weights)
    atom_weights = (atom_weights - min_value) / (max_value - min_value)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.28)
    cmap = cm.get_cmap('Oranges')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {i: plt_colors.to_rgba(atom_weights[i]) for i in range(len(nodes))}

    mol = Chem.MolFromSmiles(sml)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(6)
    op = drawer.drawOptions()

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, highlightAtoms=range(len(nodes)),
                        highlightBonds=[],
                        highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')

    with open('./graphs/'+str(id)+'.svg', 'w') as f:
        f.write(svg)


# -------------------------------------------------------------------------------------------
if __name__ == '__main__':

    num_features = 4
    num_labels = 1
    feature_str = 'h'

    FilePath = "/lustre/home/qfchen/Mult-Target-Molecular/TAGCN-amide-feature/dataset/amide_test_lbl.scv"
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    data = Utility.LoadGaoData(FilePath, num_features, feature_str, device)

    solubNet = model.GCNNet(num_features, num_labels, feature_str)

    project_path = os.getcwd()
    model_path = project_path + '/models/solubNet.pt'
    solubNet.load_state_dict(th.load(model_path, map_location='cpu'))

    print("load success")
    print('-' * 50)

    res = []
    for i, gx in enumerate(data):
        sml = data[i][0]
        y_true = gx[2]
        y_pred = th.trunc(th.sum(solubNet(gx[1]), dim=0)).item()
        print("%5d %15.1f %15.1f" % (i, y_true, y_pred))

        # Start explaining molecules_atoms
        graph = data[i][1]
        R = LRPModel(solubNet)(data[i][1])
        visualization(graph, R[-1], i, sml)