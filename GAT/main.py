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

data_file = "dataset/ESOL.csv"
all_data = Utility.LoadGaoData(data_file, num_features, feature_str, device)
random.seed(42)
random.shuffle(all_data)

ratio = 0.9

num = len(all_data)
data_train = all_data[: round(num * ratio)]
data_test = all_data[round(num * ratio):]

solubNet = model.GCNNet(num_features, num_labels, feature_str)
Utility.Train(solubNet, data_train, data_test, learning_rate, batch_size, max_epochs, output_freq, device)
