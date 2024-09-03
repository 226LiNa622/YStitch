# -*- coding: utf-8 -*-
"""

这部分是对BPNN预测模型在训练集进行10折交叉验证的代码
10-fold cross validation of BPNN prediction models in the training set
@author: Li Na and Hexiao Ding  ==  Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.setrecursionlimit(10000)  
from sklearn.model_selection import RepeatedStratifiedKFold
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Set working path
work_path = 'D:/Dataset'
os.chdir(work_path)

class BPNet(nn.Module):
    def __init__(self, input_num, hide_num, output_num):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_num, hide_num)
        self.fc2 = nn.Linear(hide_num, output_num)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def load_data(file_path):
    data = pd.read_excel(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(data)
    new_data = pd.DataFrame(scaler.transform(data))
    X = new_data.iloc[:, :-1].values.astype(np.float32)
    y = new_data.iloc[:, -1].values.astype(np.float32)
    return X, y

# origin
if __name__ == "__main__":
    file_path = "DATA_Wave1_train.xlsx"
    X, y = load_data(file_path)

    epochs = 250
    learning_rate = 0.1
    input_nodes = 25
    hidden_nodes = 6
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=0)
    results_list_BP = []
    prob_list = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_tensor = torch.FloatTensor(X[train_idx])
        y_train_tensor = torch.FloatTensor(y[train_idx])
        X_val_tensor = torch.FloatTensor(X[val_idx])
        y_val_tensor = torch.FloatTensor(y[val_idx])

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = BPNet(input_nodes, hidden_nodes, output_nodes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            scheduler.step(val_loss)

        y_pred_class = (val_outputs >= 0.5).float()
        #confusion matrix
        cm = confusion_matrix(y_val_tensor.numpy(), y_pred_class.numpy())
        #calculate
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        acc = accuracy_score(y_val_tensor.numpy(), y_pred_class.numpy())
        auc = roc_auc_score(y_val_tensor.numpy(), val_outputs.numpy())
        f_score = f1_score(y_val_tensor.numpy(), y_pred_class.numpy())
        c_index = concordance_index(y_val_tensor.numpy(), val_outputs.numpy())
        
        results_list_BP.append({
            "Fold": fold,
            "ACC": acc,
            "AUC": auc,
            "F-SCORE": f_score,
            "C-index": c_index,
            "sensitivity": sensitivity,
            "specificity": specificity
        })
        for pred, resp in zip(val_outputs.numpy(), y_val_tensor.numpy()):
            prob_list.append({
                "Fold": fold,
                "pred": pred,
                "response": resp
            })

    results_BP = pd.DataFrame(results_list_BP)
    print(results_BP.mean())  
    results_BP.to_excel("Kfold_origin_BP.xlsx", index=False)
    prob_BP = pd.DataFrame(prob_list)
    prob_BP.to_csv("Kfold-prob-origin-BP.csv", index=True)

# SHAP
if __name__ == "__main__":
    file_path = "DATA_Wave1_train_SHAP.xlsx"
    X, y = load_data(file_path)

    epochs = 200
    learning_rate = 0.1
    input_nodes = 6
    hidden_nodes = 3
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=0)
    results_list_BP = []
    prob_list = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_tensor = torch.FloatTensor(X[train_idx])
        y_train_tensor = torch.FloatTensor(y[train_idx])
        X_val_tensor = torch.FloatTensor(X[val_idx])
        y_val_tensor = torch.FloatTensor(y[val_idx])

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = BPNet(input_nodes, hidden_nodes, output_nodes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            scheduler.step(val_loss)

        y_pred_class = (val_outputs >= 0.5).float()
        #confusion matrix
        cm = confusion_matrix(y_val_tensor.numpy(), y_pred_class.numpy())
        #calculate
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        acc = accuracy_score(y_val_tensor.numpy(), y_pred_class.numpy())
        auc = roc_auc_score(y_val_tensor.numpy(), val_outputs.numpy())
        f_score = f1_score(y_val_tensor.numpy(), y_pred_class.numpy())
        c_index = concordance_index(y_val_tensor.numpy(), val_outputs.numpy())
        
        results_list_BP.append({
            "Fold": fold,
            "ACC": acc,
            "AUC": auc,
            "F-SCORE": f_score,
            "C-index": c_index,
            "sensitivity": sensitivity,
            "specificity": specificity
        })
        for pred, resp in zip(val_outputs.numpy(), y_val_tensor.numpy()):
            prob_list.append({
                "Fold": fold,
                "pred": pred,
                "response": resp
            })

    results_BP = pd.DataFrame(results_list_BP)
    print(results_BP.mean())  
    results_BP.to_excel("Kfold_SHAP_BP.xlsx", index=False)
    prob_BP = pd.DataFrame(prob_list)
    prob_BP.to_csv("Kfold-prob-SHAP-BP.csv", index=True)

# lasso
if __name__ == "__main__":
    file_path = "DATA_Wave1_train_lasso.xlsx"
    X, y = load_data(file_path)

    epochs = 200
    learning_rate = 0.1
    input_nodes = 22
    hidden_nodes = 6
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=0)
    results_list_BP = []
    prob_list = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_tensor = torch.FloatTensor(X[train_idx])
        y_train_tensor = torch.FloatTensor(y[train_idx])
        X_val_tensor = torch.FloatTensor(X[val_idx])
        y_val_tensor = torch.FloatTensor(y[val_idx])

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = BPNet(input_nodes, hidden_nodes, output_nodes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            scheduler.step(val_loss)

        y_pred_class = (val_outputs >= 0.5).float()
        #confusion matrix
        cm = confusion_matrix(y_val_tensor.numpy(), y_pred_class.numpy())
        #calculate
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        acc = accuracy_score(y_val_tensor.numpy(), y_pred_class.numpy())
        auc = roc_auc_score(y_val_tensor.numpy(), val_outputs.numpy())
        f_score = f1_score(y_val_tensor.numpy(), y_pred_class.numpy())
        c_index = concordance_index(y_val_tensor.numpy(), val_outputs.numpy())
        
        results_list_BP.append({
            "Fold": fold,
            "ACC": acc,
            "AUC": auc,
            "F-SCORE": f_score,
            "C-index": c_index,
            "sensitivity": sensitivity,
            "specificity": specificity
        })
        for pred, resp in zip(val_outputs.numpy(), y_val_tensor.numpy()):
            prob_list.append({
                "Fold": fold,
                "pred": pred,
                "response": resp
            })

    results_BP = pd.DataFrame(results_list_BP)
    print(results_BP.mean())  
    results_BP.to_excel("Kfold_lasso_BP.xlsx", index=False)
    prob_BP = pd.DataFrame(prob_list)
    prob_BP.to_csv("Kfold-prob-lasso-BP.csv", index=True)
    
# VIMP-gain
if __name__ == "__main__":
    file_path = "DATA_Wave1_train_gain.xlsx"
    X, y = load_data(file_path)

    epochs = 200
    learning_rate = 0.1
    input_nodes = 5
    hidden_nodes = 2
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=0)
    results_list_BP = []
    prob_list = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_tensor = torch.FloatTensor(X[train_idx])
        y_train_tensor = torch.FloatTensor(y[train_idx])
        X_val_tensor = torch.FloatTensor(X[val_idx])
        y_val_tensor = torch.FloatTensor(y[val_idx])

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = BPNet(input_nodes, hidden_nodes, output_nodes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            scheduler.step(val_loss)

        y_pred_class = (val_outputs >= 0.5).float()
        #confusion matrix
        cm = confusion_matrix(y_val_tensor.numpy(), y_pred_class.numpy())
        #calculate
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        acc = accuracy_score(y_val_tensor.numpy(), y_pred_class.numpy())
        auc = roc_auc_score(y_val_tensor.numpy(), val_outputs.numpy())
        f_score = f1_score(y_val_tensor.numpy(), y_pred_class.numpy())
        c_index = concordance_index(y_val_tensor.numpy(), val_outputs.numpy())
        
        results_list_BP.append({
            "Fold": fold,
            "ACC": acc,
            "AUC": auc,
            "F-SCORE": f_score,
            "C-index": c_index,
            "sensitivity": sensitivity,
            "specificity": specificity
        })
        for pred, resp in zip(val_outputs.numpy(), y_val_tensor.numpy()):
            prob_list.append({
                "Fold": fold,
                "pred": pred,
                "response": resp
            })

    results_BP = pd.DataFrame(results_list_BP)
    print(results_BP.mean())  
    results_BP.to_excel("Kfold_gain_BP.xlsx", index=False)
    prob_BP = pd.DataFrame(prob_list)
    prob_BP.to_csv("Kfold-prob-gain-BP.csv", index=True)
    
# VIMP-weight
if __name__ == "__main__":
    file_path = "DATA_Wave1_train_weight.xlsx"
    X, y = load_data(file_path)

    epochs = 200
    learning_rate = 0.08
    input_nodes = 8
    hidden_nodes = 3
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8)

    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=0)
    results_list_BP = []
    prob_list = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_tensor = torch.FloatTensor(X[train_idx])
        y_train_tensor = torch.FloatTensor(y[train_idx])
        X_val_tensor = torch.FloatTensor(X[val_idx])
        y_val_tensor = torch.FloatTensor(y[val_idx])

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = BPNet(input_nodes, hidden_nodes, output_nodes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            scheduler.step(val_loss)

        y_pred_class = (val_outputs >= 0.5).float()
        #confusion matrix
        cm = confusion_matrix(y_val_tensor.numpy(), y_pred_class.numpy())
        #calculate
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        acc = accuracy_score(y_val_tensor.numpy(), y_pred_class.numpy())
        auc = roc_auc_score(y_val_tensor.numpy(), val_outputs.numpy())
        f_score = f1_score(y_val_tensor.numpy(), y_pred_class.numpy())
        c_index = concordance_index(y_val_tensor.numpy(), val_outputs.numpy())
        
        results_list_BP.append({
            "Fold": fold,
            "ACC": acc,
            "AUC": auc,
            "F-SCORE": f_score,
            "C-index": c_index,
            "sensitivity": sensitivity,
            "specificity": specificity
        })
        for pred, resp in zip(val_outputs.numpy(), y_val_tensor.numpy()):
            prob_list.append({
                "Fold": fold,
                "pred": pred,
                "response": resp
            })

    results_BP = pd.DataFrame(results_list_BP)
    print(results_BP.mean())  
    results_BP.to_excel("Kfold_weight_BP.xlsx", index=False)
    prob_BP = pd.DataFrame(prob_list)
    prob_BP.to_csv("Kfold-prob-weight-BP.csv", index=True)

# VIMP-cover
if __name__ == "__main__":
    file_path = "DATA_Wave1_train_cover.xlsx"
    X, y = load_data(file_path)

    epochs = 200
    learning_rate = 0.1
    input_nodes = 4
    hidden_nodes = 2
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=0)
    results_list_BP = []
    prob_list = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_tensor = torch.FloatTensor(X[train_idx])
        y_train_tensor = torch.FloatTensor(y[train_idx])
        X_val_tensor = torch.FloatTensor(X[val_idx])
        y_val_tensor = torch.FloatTensor(y[val_idx])

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = BPNet(input_nodes, hidden_nodes, output_nodes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            scheduler.step(val_loss)

        y_pred_class = (val_outputs >= 0.5).float()
        #confusion matrix
        cm = confusion_matrix(y_val_tensor.numpy(), y_pred_class.numpy())
        #calculate
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        acc = accuracy_score(y_val_tensor.numpy(), y_pred_class.numpy())
        auc = roc_auc_score(y_val_tensor.numpy(), val_outputs.numpy())
        f_score = f1_score(y_val_tensor.numpy(), y_pred_class.numpy())
        c_index = concordance_index(y_val_tensor.numpy(), val_outputs.numpy())
        
        results_list_BP.append({
            "Fold": fold,
            "ACC": acc,
            "AUC": auc,
            "F-SCORE": f_score,
            "C-index": c_index,
            "sensitivity": sensitivity,
            "specificity": specificity
        })
        for pred, resp in zip(val_outputs.numpy(), y_val_tensor.numpy()):
            prob_list.append({
                "Fold": fold,
                "pred": pred,
                "response": resp
            })

    results_BP = pd.DataFrame(results_list_BP)
    print(results_BP.mean())  
    results_BP.to_excel("Kfold_cover_BP.xlsx", index=False)
    prob_BP = pd.DataFrame(prob_list)
    prob_BP.to_csv("Kfold-prob-cover-BP.csv", index=True)
    
