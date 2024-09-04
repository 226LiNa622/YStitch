# -*- coding: utf-8 -*-
"""

这部分是预测模型最佳参数调整的代码
Optimal parameter adjustment of prediction model
@author: Li Na and Hexiao Ding  ==  Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# Set working path
work_path = 'D:/Dataset'
os.chdir(work_path)
# Min-Max normalization
def scal_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(data)
    new_data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    x = new_data.iloc[:, :-1]
    y = new_data.iloc[:, -1]
    return x, y
# Input Data
# 1.Without prunning
new_data1 = pd.read_excel("DATA_Wave1_train.xlsx")
x1, y1 = scal_data(new_data1)
# 2.prunning with SHAP
new_data2 = pd.read_excel("DATA_Wave1_train_SHAP.xlsx")
x2, y2 = scal_data(new_data2)
# 3.prunning with Lasso
new_data3 = pd.read_excel("DATA_Wave1_train_lasso.xlsx")
x3, y3 = scal_data(new_data3)
# 4.prunning with VIMP-gain
new_data4 = pd.read_excel("DATA_Wave1_train_gain.xlsx")
x4, y4 = scal_data(new_data4)
# 5.prunning with VIMP-weight
new_data5 = pd.read_excel("DATA_Wave1_train_weight.xlsx")
x5, y5 = scal_data(new_data5)
# 6.prunning with VIMP-cover
new_data6 = pd.read_excel("DATA_Wave1_train_cover.xlsx")
x6, y6 = scal_data(new_data6)


###############################################################################
###############################################################################
##############  Adaboost   调参///  Turning Stage of Adaboost  ################

def ada_turn(x, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    # Turning-AdaBoost
    AdaBoost=ensemble.AdaBoostClassifier(random_state=0)
    param_grid_ada = {
        'n_estimators': np.linspace(50, 150, 2).astype(int),
        'learning_rate': np.linspace(0.01, 1.0, 5)
    }
    grid_search_ada = GridSearchCV(AdaBoost, param_grid=param_grid_ada, scoring="f1", cv=cv, n_jobs=-1)
    grid_search_ada.fit(x, y) 
    return grid_search_ada.best_params_

# x1 y1
best_params_1 = ada_turn(x1, y1) 
print("AdaBoost's optimum parameter:", best_params_1)
# AdaBoost's optimum parameter： {'learning_rate': 0.5, 'n_estimators': 116}

# x2 y2
best_params_2 = ada_turn(x2, y2) 
print("AdaBoost's optimum parameter:", best_params_2)
# AdaBoost's optimum parameter: {'learning_rate': 0.64, 'n_estimators': 75}

# x3 y3
best_params_3 = ada_turn(x3, y3) 
print("AdaBoost's optimum parameter:", best_params_3)
# AdaBoost's optimum parameter: {'learning_rate': 0.76, 'n_estimators': 113}

# x4 y4
best_params_4 = ada_turn(x4, y4) 
print("AdaBoost's optimum parameter:", best_params_4)
# AdaBoost's optimum parameter: {'learning_rate': 0.86, 'n_estimators': 120}

# x5 y5
best_params_5 = ada_turn(x5, y5) 
print("AdaBoost's optimum parameter:", best_params_5)
#AdaBoost's optimum parameter: {'learning_rate': 0.39, 'n_estimators': 50}

# x6 y6
best_params_6 = ada_turn(x6, y6) 
print("AdaBoost's optimum parameter:", best_params_6)
#AdaBoost's optimum parameter: {'learning_rate': 0.5800000000000001, 'n_estimators': 91}


###############################################################################
###############################################################################
###  Logistic Regression  调参///  Turning Stage of Logistic Regression  ######

from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

def lr_turn(x, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    # Turning-LR
    LR=LogisticRegression(random_state=0)
    param_grid_lr = {
        'max_iter': np.linspace(100, 1000, 100).astype(int)
    }
    grid_search_lr = GridSearchCV(LR, param_grid_lr, scoring="f1", cv=cv,n_jobs=-1)
    grid_search_lr.fit(x, y) 
    return grid_search_lr.best_params_

# x1 y1
best_paramsl = lr_turn(x1, y1) 
print("LR最佳参数组合：", best_paramsl)
# x2 y2
best_params2 = lr_turn(x2, y2) 
print("LR最佳参数组合：", best_params2)
# x3 y3
best_params3 = lr_turn(x3, y3) 
print("LR最佳参数组合：", best_params3)
# x4 y4
best_params4 = lr_turn(x4, y4) 
print("LR最佳参数组合：", best_params4)
# x5 y5
best_params5 = lr_turn(x5, y5) 
print("LR最佳参数组合：", best_params5)
# x6 y6
best_params6 = lr_turn(x6, y6) 
print("LR最佳参数组合：", best_params6)

# Without prunning: {'max_iter': 100}
# prunning with SHAP: {'max_iter': 100}
# prunning with Lasso:{'max_iter': 100}
# prunning with VIMP-gain:{'max_iter': 100}
# prunning with VIMP-weight:{'max_iter': 100}
# prunning with VIMP-cover:{'max_iter': 100}


###############################################################################
###############################################################################
#################  神经网络调参///  Turning Stage of BPNN  #####################
#################  本步骤手动二分法调参，最终结果即主函数参数  ####################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# Define the BP neural network model
class BPNet(nn.Module):
    def __init__(self, input_num, hide_num, output_num):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_num, hide_num)
        self.fc2 = nn.Linear(hide_num, output_num)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Train and validate the model
def train_fold(X_train, y_train, X_val, y_val, model, criterion, optimizer, epochs, scheduler):
    train_losses, val_losses, train_f1_scores, val_f1_scores = [], [], [], []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))  
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Calculate F1 Score
        y_pred = (outputs > 0.5).float()  # Threshold decision
        train_f1_scores.append(f1_score(y_train, y_pred.numpy(), average='macro'))
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.view(-1, 1))
            val_losses.append(val_loss.item())
            
            y_val_pred = (val_outputs > 0.5).float()  # Threshold decision
            val_f1_scores.append(f1_score(y_val, y_val_pred.numpy(), average='macro'))
        
        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch+1}, Current learning rate is: {param_group['lr']}")
    
    return train_losses, val_losses, train_f1_scores, val_f1_scores

# origin
if __name__ == "__main__":
    # parameter
    epochs = 250  
    learning_rate = 0.1  # Initial learning rate
    input_nodes = 25  # Input layer = Number of features
    hidden_nodes = 6  # The number of hidden layers need to be manually adjusted
    output_nodes = 1  
    # dataset
    x1 = x1.values.astype(np.float32)
    y1 = y1.values.astype(np.float32)
    # model
    model = BPNet(input_nodes, hidden_nodes, output_nodes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results1 = []

    for train_idx, val_idx in skf.split(x1, y1):
        X_train_tensor = torch.FloatTensor(x1[train_idx])
        y_train_tensor = torch.FloatTensor(y1[train_idx])
        X_val_tensor = torch.FloatTensor(x1[val_idx])
        y_val_tensor = torch.FloatTensor(y1[val_idx])

        train_losses, val_losses, train_f1, val_f1 = train_fold(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, model, criterion, optimizer, epochs, scheduler)
        results1.append((train_losses, val_losses, train_f1, val_f1))

# Calculate the average of the metrics
def average_metrics(results1):
    avg_train_losses1 = np.mean([res[0] for res in results1], axis=0)
    avg_val_losses1 = np.mean([res[1] for res in results1], axis=0)
    avg_train_f11 = np.mean([res[2] for res in results1], axis=0)
    avg_val_f11 = np.mean([res[3] for res in results1], axis=0)
    return avg_train_losses1, avg_val_losses1, avg_train_f11, avg_val_f11

avg_train_losses1, avg_val_losses1, avg_train_f11, avg_val_f11 = average_metrics(results1)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), avg_train_losses1, label='Average Training Loss')
plt.plot(range(1, epochs+1), avg_val_losses1, label='Average Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), avg_train_f11, label='Average Training F1 Score')
plt.plot(range(1, epochs+1), avg_val_f11, label='Average Validation F1 Score')
plt.title('Training and Validation F1 Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

# SHAP
if __name__ == "__main__":
    # parameter
    epochs = 200  
    learning_rate = 0.1  # Initial learning rate
    input_nodes = 6  # Input layer = Number of features
    hidden_nodes = 3 # The number of hidden layers need to be manually adjusted
    output_nodes = 1  
    # dataset
    x2 = x2.values.astype(np.float32)
    y2 = y2.values.astype(np.float32)
    # model
    model = BPNet(input_nodes, hidden_nodes, output_nodes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results2 = []

    for train_idx, val_idx in skf.split(x2, y2):
        X_train_tensor = torch.FloatTensor(x2[train_idx])
        y_train_tensor = torch.FloatTensor(y2[train_idx])
        X_val_tensor = torch.FloatTensor(x2[val_idx])
        y_val_tensor = torch.FloatTensor(y2[val_idx])

        train_losses, val_losses, train_f1, val_f1 = train_fold(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, model, criterion, optimizer, epochs, scheduler)
        results2.append((train_losses, val_losses, train_f1, val_f1))

# Calculate the average of the metrics
def average_metrics(results2):
    avg_train_losses2 = np.mean([res[0] for res in results2], axis=0)
    avg_val_losses2 = np.mean([res[1] for res in results2], axis=0)
    avg_train_f12 = np.mean([res[2] for res in results2], axis=0)
    avg_val_f12 = np.mean([res[3] for res in results2], axis=0)
    return avg_train_losses2, avg_val_losses2, avg_train_f12, avg_val_f12

avg_train_losses2, avg_val_losses2, avg_train_f12, avg_val_f12 = average_metrics(results2)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), avg_train_losses2, label='Average Training Loss')
plt.plot(range(1, epochs+1), avg_val_losses2, label='Average Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), avg_train_f12, label='Average Training F1 Score')
plt.plot(range(1, epochs+1), avg_val_f12, label='Average Validation F1 Score')
plt.title('Training and Validation F1 Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

# lasso
if __name__ == "__main__":
    # parameter
    epochs = 200 
    learning_rate = 0.1  # Initial learning rate
    input_nodes = 22  # Input layer = Number of features
    hidden_nodes = 6  # The number of hidden layers need to be manually adjusted
    output_nodes = 1  
    # dataset
    x3 = x3.values.astype(np.float32)
    y3 = y3.values.astype(np.float32)
    # model
    model = BPNet(input_nodes, hidden_nodes, output_nodes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results3 = []

    for train_idx, val_idx in skf.split(x3, y3):
        X_train_tensor = torch.FloatTensor(x3[train_idx])
        y_train_tensor = torch.FloatTensor(y3[train_idx])
        X_val_tensor = torch.FloatTensor(x3[val_idx])
        y_val_tensor = torch.FloatTensor(y3[val_idx])

        train_losses, val_losses, train_f1, val_f1 = train_fold(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, model, criterion, optimizer, epochs, scheduler)
        results3.append((train_losses, val_losses, train_f1, val_f1))

# Calculate the average of the metrics
def average_metrics(results3):
    avg_train_losses3 = np.mean([res[0] for res in results3], axis=0)
    avg_val_losses3 = np.mean([res[1] for res in results3], axis=0)
    avg_train_f13 = np.mean([res[2] for res in results3], axis=0)
    avg_val_f13 = np.mean([res[3] for res in results3], axis=0)
    return avg_train_losses3, avg_val_losses3, avg_train_f13, avg_val_f13

avg_train_losses3, avg_val_losses3, avg_train_f13, avg_val_f13 = average_metrics(results3)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), avg_train_losses3, label='Average Training Loss')
plt.plot(range(1, epochs+1), avg_val_losses3, label='Average Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), avg_train_f13, label='Average Training F1 Score')
plt.plot(range(1, epochs+1), avg_val_f13, label='Average Validation F1 Score')
plt.title('Training and Validation F1 Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

# VIMP-gain
if __name__ == "__main__":
    # parameter
    epochs = 200  
    learning_rate = 0.1  # Initial learning rate
    input_nodes = 5  # Input layer = Number of features
    hidden_nodes = 2  # The number of hidden layers need to be manually adjusted
    output_nodes = 1  
    # dataset
    x4 = x4.values.astype(np.float32)
    y4 = y4.values.astype(np.float32)
    # model
    model = BPNet(input_nodes, hidden_nodes, output_nodes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results4 = []

    for train_idx, val_idx in skf.split(x4, y4):
        X_train_tensor = torch.FloatTensor(x4[train_idx])
        y_train_tensor = torch.FloatTensor(y4[train_idx])
        X_val_tensor = torch.FloatTensor(x4[val_idx])
        y_val_tensor = torch.FloatTensor(y4[val_idx])

        train_losses, val_losses, train_f1, val_f1 = train_fold(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, model, criterion, optimizer, epochs, scheduler)
        results4.append((train_losses, val_losses, train_f1, val_f1))

# Calculate the average of the metrics
def average_metrics(results4):
    avg_train_losses4 = np.mean([res[0] for res in results4], axis=0)
    avg_val_losses4 = np.mean([res[1] for res in results4], axis=0)
    avg_train_f14 = np.mean([res[2] for res in results4], axis=0)
    avg_val_f14 = np.mean([res[3] for res in results4], axis=0)
    return avg_train_losses4, avg_val_losses4, avg_train_f14, avg_val_f14

avg_train_losses4, avg_val_losses4, avg_train_f14, avg_val_f14 = average_metrics(results4)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), avg_train_losses4, label='Average Training Loss')
plt.plot(range(1, epochs+1), avg_val_losses4, label='Average Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), avg_train_f14, label='Average Training F1 Score')
plt.plot(range(1, epochs+1), avg_val_f14, label='Average Validation F1 Score')
plt.title('Training and Validation F1 Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

# VIMP-weight
if __name__ == "__main__":
    # parameter
    epochs = 200  
    learning_rate = 0.08  # Initial learning rate
    input_nodes = 8  # Input layer = Number of features
    hidden_nodes = 3  # The number of hidden layers need to be manually adjusted
    output_nodes = 1  
    # dataset
    x5 = x5.values.astype(np.float32)
    y5 = y5.values.astype(np.float32)
    # model
    model = BPNet(input_nodes, hidden_nodes, output_nodes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results5 = []

    for train_idx, val_idx in skf.split(x5, y5):
        X_train_tensor = torch.FloatTensor(x5[train_idx])
        y_train_tensor = torch.FloatTensor(y5[train_idx])
        X_val_tensor = torch.FloatTensor(x5[val_idx])
        y_val_tensor = torch.FloatTensor(y5[val_idx])

        train_losses, val_losses, train_f1, val_f1 = train_fold(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, model, criterion, optimizer, epochs, scheduler)
        results5.append((train_losses, val_losses, train_f1, val_f1))

# Calculate the average of the metrics
def average_metrics(results5):
    avg_train_losses5 = np.mean([res[0] for res in results5], axis=0)
    avg_val_losses5 = np.mean([res[1] for res in results5], axis=0)
    avg_train_f15 = np.mean([res[2] for res in results5], axis=0)
    avg_val_f15 = np.mean([res[3] for res in results5], axis=0)
    return avg_train_losses5, avg_val_losses5, avg_train_f15, avg_val_f15

avg_train_losses5, avg_val_losses5, avg_train_f15, avg_val_f15 = average_metrics(results5)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), avg_train_losses5, label='Average Training Loss')
plt.plot(range(1, epochs+1), avg_val_losses5, label='Average Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), avg_train_f15, label='Average Training F1 Score')
plt.plot(range(1, epochs+1), avg_val_f15, label='Average Validation F1 Score')
plt.title('Training and Validation F1 Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

# VIMP-cover
if __name__ == "__main__":
    # parameter
    epochs = 200  
    learning_rate = 0.1  # Initial learning rate
    input_nodes = 4  # Input layer = Number of features
    hidden_nodes = 2  # The number of hidden layers need to be manually adjusted
    output_nodes = 1  
    # dataset
    x6 = x6.values.astype(np.float32)
    y6 = y6.values.astype(np.float32)
    # model
    model = BPNet(input_nodes, hidden_nodes, output_nodes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results6 = []

    for train_idx, val_idx in skf.split(x6, y6):
        X_train_tensor = torch.FloatTensor(x6[train_idx])
        y_train_tensor = torch.FloatTensor(y6[train_idx])
        X_val_tensor = torch.FloatTensor(x6[val_idx])
        y_val_tensor = torch.FloatTensor(y6[val_idx])

        train_losses, val_losses, train_f1, val_f1 = train_fold(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, model, criterion, optimizer, epochs, scheduler)
        results6.append((train_losses, val_losses, train_f1, val_f1))

# Calculate the average of the metrics
def average_metrics(results6):
    avg_train_losses6 = np.mean([res[0] for res in results6], axis=0)
    avg_val_losses6 = np.mean([res[1] for res in results6], axis=0)
    avg_train_f16 = np.mean([res[2] for res in results6], axis=0)
    avg_val_f16 = np.mean([res[3] for res in results6], axis=0)
    return avg_train_losses6, avg_val_losses6, avg_train_f16, avg_val_f16

avg_train_losses6, avg_val_losses6, avg_train_f16, avg_val_f16 = average_metrics(results6)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), avg_train_losses6, label='Average Training Loss')
plt.plot(range(1, epochs+1), avg_val_losses6, label='Average Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), avg_train_f16, label='Average Training F1 Score')
plt.plot(range(1, epochs+1), avg_val_f16, label='Average Validation F1 Score')
plt.title('Training and Validation F1 Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()
