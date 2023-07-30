# Author QFIUNE
# coding=utf-8
# @Time: 2023/3/30 14:10
# @File: ConvertSDF.py
# @Software: PyCharm
# @contact: 1760812842@qq.com

import sys
from rdkit import Chem


def converter(file_name):
    mols = [mol for mol in Chem.SDMolSupplier(file_name) if mol]
    outname = file_name.split(".sdf")[0] + ".csv"
    out_file = open(outname, "w")
    for mol in mols:
        smi = Chem.MolToSmiles(mol)
        name = mol.GetProp("logS")
        out_file.write("{},{}\n".format(smi, name))
    out_file.close()


if __name__=="__main__":
    file = "./Hou/data_set.sdf"
    converter(file)