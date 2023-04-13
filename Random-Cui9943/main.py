import random
import pandas as pd
import torch as th
from mtMolDes import model, Utility
from mtMolDes.Evaluation import MAE, MSE, RMSE, Spearman
from sklearn.metrics import r2_score
import numpy as np
import os

def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True
setup_seed(42)


num_features = 4
num_labels = 1
feature_str = "h"

batch_size = 32
learning_rate = 1e-3
max_epochs = 500
output_freq = 1

# device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
device = th.device("cpu")


data_file = "dataset/Cui9943.csv"
all_data = Utility.LoadGaoData(data_file, num_features, feature_str, device)
print("# of all graphs/labels:      %d" % (len(all_data)))

random.seed(42)
random.shuffle(all_data)

ratio_1 = 0.8
ratio_2 = 0.9

num = len(all_data)
data_train = all_data[: round(num * ratio_1)]
data_valid = all_data[round(num * ratio_1): round(num * ratio_2)]
data_test = all_data[round(num * ratio_2):]

print("# of all graphs/labels:      %d" % (len(data_train)+len(data_valid)+len(data_test)))
print("# of training graphs/labels:      %d" % (len(data_train)))
print("# of validing graphs/labels:      %d" % (len(data_valid)))
print("# of testing graphs/labels:      %d" % (len(data_test)))

TrainDataset = [[gx[0], gx[2]] for gx in data_train]
ValidDataset = [[gx[0], gx[2]] for gx in data_valid]
TestDataset = [[gx[0], gx[2]] for gx in data_test]
name = ['smiles', 'LogS']

df_train = pd.DataFrame(columns=name, data=TrainDataset)
df_valid = pd.DataFrame(columns=name, data=ValidDataset)
df_test = pd.DataFrame(columns=name, data=TestDataset)

df_train.to_csv('dataset/CuiTrain.csv', index=False)
df_valid.to_csv('dataset/CuiValid.csv', index=False)
df_test.to_csv('dataset/CuiTest.csv', index=False)


solubNet = model.GCNNet(num_features, num_labels, feature_str)
Utility.Train(solubNet, data_train, data_valid, learning_rate, batch_size, max_epochs, output_freq, device)

project_path = os.getcwd()
model_path = project_path + '/solubNet.pt'
solubNet.load_state_dict(th.load(model_path, map_location='cpu'))
