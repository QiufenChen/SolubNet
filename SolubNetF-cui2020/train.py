import random
import pandas as pd
import torch as th
from mtMolDes import model, Utility
from mtMolDes.Evaluation import MAE, MSE, RMSE
import numpy as np

num_features = 4
num_labels = 1
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
feature_str = "h"

data_file = "dataset/new_cui2020.csv"

all_data = Utility.LoadGaoData(data_file, num_features, feature_str, device)
print("# of all graphs/labels:      %d" % (len(all_data)))

random.seed(1023)
random.shuffle(all_data)

ratio = 0.8
num = len(all_data)

data_train = all_data[: round(num * ratio)]
data_test = all_data[round(num * ratio):]
print("# of testing graphs/labels:      %d" % (len(data_test)))
solubNet = model.GCNNet(num_features, num_labels, feature_str)

training_ratio = 0.8
batch_size = 64
learning_rate = 1.e-3
max_epochs = 1000
output_freq = 1
save_fn_prefix = "models/solub"

Utility.Train(solubNet, data_train, learning_rate, batch_size, max_epochs, output_freq, save_fn_prefix, device)


test = []
y_pred = []
y_ture = []
solubNet.eval()
for i, gx in enumerate(data_test):
    true_prop = round(gx[2], 3)
    pred_prop = round(th.sum(solubNet(gx[1]), dim=0).item(), 3)
    print("%5d %15.8f %15.8f" % (i, true_prop, pred_prop))

    test.append([gx[0], true_prop, pred_prop])
    y_pred.append(pred_prop)
    y_ture.append(true_prop)
    
y_pred = th.from_numpy(np.array(y_pred))
y_ture = th.from_numpy(np.array(y_ture))

name = ['structure', 'true-value', 'pred-vaule']
df = pd.DataFrame(columns=name, data=test)
df.to_csv('dataset/test.csv', index=False)

test_mae = MAE(y_pred, y_ture)
test_mse = MSE(y_pred, y_ture)
test_rmse = RMSE(y_pred, y_ture)

print("%15.7f %15.7f %15.7f" % (test_mae, test_mse, test_rmse))

