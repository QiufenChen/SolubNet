import random
import pandas as pd
import torch as th
from mtMolDes import model, Utility
from mtMolDes.Evaluation import MAE, MSE, RMSE, Spearman
from sklearn.metrics import r2_score
import numpy as np

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

TrainFile = "dataset/CuiTrain.csv"
ValidFile = "dataset/CuiValid.csv"
TrainData = Utility.LoadGaoData(TrainFile, num_features, feature_str, device)
ValidData = Utility.LoadGaoData(ValidFile, num_features, feature_str, device)

solubNet = model.GCNNet(num_features, num_labels, feature_str)
Utility.Train(solubNet, TrainData, ValidData, learning_rate, batch_size, max_epochs, output_freq, device)







# random.seed(1022)
# random.shuffle(all_data)
# ratio = 0.9
# num = len(all_data)

# data_train = all_data[: round(num * ratio)]
# data_Valid = all_data[round(num * ratio):]
# print("# of Validing graphs/labels:      %d" % (len(data_Valid)))

# Valid = []
# y_pred = []
# y_ture = []
# solubNet.eval()
# for i, gx in enumerate(data_Valid):
#     true_prop = round(gx[2], 3)
#     pred_prop = round(th.sum(solubNet(gx[1]), dim=0).item(), 3)
#     print("%5d %15.8f %15.8f" % (i, true_prop, pred_prop))

#     Valid.append([gx[0], true_prop, pred_prop])
#     y_pred.append(pred_prop)
#     y_ture.append(true_prop)
    
# y_pred = th.from_numpy(np.array(y_pred))
# y_ture = th.from_numpy(np.array(y_ture))

# name = ['structure', 'true-value', 'pred-vaule']
# df = pd.DataFrame(columns=name, data=Valid)
# df.to_csv('dataset/Valid.csv', index=False)

# Valid_mae = MAE(y_pred, y_ture)
# Valid_mse = MSE(y_pred, y_ture)
# Valid_rmse = RMSE(y_pred, y_ture)
# Valid_r2 = r2_score(y_pred.detach().numpy(), y_ture.detach().numpy())
# Valid_cc = Spearman(y_pred.detach().numpy(), y_ture.detach().numpy())


# print("%15s %15s %15s %15s %15s" %
#           ("ValidMAE", "ValidMSE", "ValidRMSE", "ValidR2", "ValidCC"))
# print("%15.7f %15.7f %15.7f %15.7f %15.7f" % (Valid_mae, Valid_mse, Valid_rmse, Valid_r2, Valid_cc))

