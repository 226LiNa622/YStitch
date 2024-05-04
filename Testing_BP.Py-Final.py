# -*- coding: utf-8 -*-
"""

这部分是对测试集使用bootstrapping评价BPNN训练模型的代码（调用GPU）
Evaluate BP model using bootstrapping on the test set (Using GPU)
@author: Li Na and Hexiao Ding  ==  Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from lifelines.utils import concordance_index
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import resample

class BPNet(nn.Module):
    def __init__(self, input_num, hide_num, output_num):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_num, hide_num)
        self.fc2 = nn.Linear(hide_num, output_num)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def load_training_data(file_path_train):
    data = pd.read_excel(file_path_train)
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(data)
    new_data = pd.DataFrame(scaler.transform(data))
    X = new_data.iloc[:, :-1].values.astype(np.float32)
    y = new_data.iloc[:, -1].values.astype(np.float32)
    return X, y

def load_testing_data(file_path_test):
    data = pd.read_excel(file_path_test)
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(data)
    values = data.values
    return values

def get_results(device, X_train, y_train, values, model, criterion, optimizer, scheduler, epochs):
    results_boostrap = []
    prob_list = []
    n_bootstrap = 1000  # Number of bootstrap iterations
    
    initial_state = model.state_dict()
    
    for i in range(n_bootstrap):
        model.load_state_dict(initial_state)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        subtest = resample(values, replace=True, n_samples=int(len(values)))
        X_test = subtest[:, :-1].astype(np.float32)
        y_test = subtest[:, -1].astype(np.float32)
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_test).to(device)
        y_val_tensor = torch.FloatTensor(y_test).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

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

        y_pred_class = (val_outputs >= 0.5).float().cpu()
        # confusion matrix
        cm = confusion_matrix(y_val_tensor.cpu().numpy(), y_pred_class.numpy())
        # calculate
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        acc = accuracy_score(y_val_tensor.cpu().numpy(), y_pred_class.numpy())
        auc = roc_auc_score(y_val_tensor.cpu().numpy(), val_outputs.cpu().numpy())
        f_score = f1_score(y_val_tensor.cpu().numpy(), y_pred_class.numpy())
        c_index = concordance_index(y_val_tensor.cpu().numpy(), val_outputs.cpu().numpy())
        
        results_boostrap.append({
            "n_bootstrap": i,
            "ACC": acc,
            "AUC": auc,
            "F-SCORE": f_score,
            "C-index": c_index,
            "sensitivity": sensitivity,
            "specificity": specificity
        })
        results_boostrap_df = pd.DataFrame(results_boostrap)
        for pred, resp in zip(val_outputs.cpu().numpy(), y_val_tensor.cpu().numpy()):
            prob_list.append({
                "n_bootstrap": i,
                "pred": pred,
                "response": resp
            })
            prob_BP = pd.DataFrame(prob_list)
    return results_boostrap_df, prob_BP

# origin
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path_train = "C:/Users/Administrator/Desktop/ML/DATA_Wave1_train.xlsx"
    file_path_test = "C:/Users/Administrator/Desktop/ML/DATA_Wave1_test.xlsx"
    X_train, y_train = load_training_data(file_path_train)
    values = load_testing_data(file_path_test)

    epochs = 250
    learning_rate = 0.1
    input_nodes = 25
    hidden_nodes = 6
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes).to(device)  
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    results_boostrap_df, prob_BP = get_results(device, X_train, y_train, values, model, criterion, optimizer, scheduler, epochs)
    print(results_boostrap_df.mean())
    results_boostrap_df.to_excel("testing_origin_BP.xlsx", index=False)
    prob_BP.to_csv("testing-prob-origin-BP.csv", index=True)

# SHAP
if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    file_path_train = "C:/Users/Administrator/Desktop/ML/DATA_Wave1_train_SHAP.xlsx"
    file_path_test = "C:/Users/Administrator/Desktop/ML/DATA_Wave1_test_SHAP.xlsx"
    X_train, y_train = load_training_data(file_path_train)
    values = load_testing_data(file_path_test)
    
    epochs = 200
    learning_rate = 0.1
    input_nodes = 6
    hidden_nodes = 3
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    results_boostrap_df, prob_BP = get_results(device, X_train, y_train, values, model, criterion, optimizer, scheduler, epochs)
    print(results_boostrap_df.mean())
    results_boostrap_df.to_excel("testing_SHAP_BP.xlsx", index=False)
    prob_BP.to_csv("testing-prob-SHAP-BP.csv", index=True)

# lasso
if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    file_path_train = "C:/Users/Administrator/Desktop/ML/DATA_Wave1_train_lasso.xlsx"
    file_path_test = "C:/Users/Administrator/Desktop/ML/DATA_Wave1_test_lasso.xlsx"
    X_train, y_train = load_training_data(file_path_train)
    values = load_testing_data(file_path_test)
    
    epochs = 200
    learning_rate = 0.1
    input_nodes = 22
    hidden_nodes = 6
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes).to(device)  
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    results_boostrap_df, prob_BP = get_results(device, X_train, y_train, values, model, criterion, optimizer, scheduler, epochs)
    print(results_boostrap_df.mean())
    results_boostrap_df.to_excel("testing_lasso_BP.xlsx", index=False)
    prob_BP.to_csv("testing-prob-lasso-BP.csv", index=True)

# VIMP-gain
if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    file_path_train = "C:/Users/Administrator/Desktop/ML/DATA_Wave1_train_gain.xlsx"
    file_path_test = "C:/Users/Administrator/Desktop/ML/DATA_Wave1_test_gain.xlsx"
    X_train, y_train = load_training_data(file_path_train)
    values = load_testing_data(file_path_test)

    epochs = 200
    learning_rate = 0.1
    input_nodes = 5
    hidden_nodes = 2
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes).to(device)   
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    results_boostrap_df, prob_BP = get_results(device, X_train, y_train, values, model, criterion, optimizer, scheduler, epochs)
    print(results_boostrap_df.mean())
    results_boostrap_df.to_excel("testing_gain_BP.xlsx", index=False)
    prob_BP.to_csv("testing-prob-gain-BP.csv", index=True)

# VIMP-weight
if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    file_path_train = "C:/Users/Administrator/Desktop/ML/DATA_Wave1_train_weight.xlsx"
    file_path_test = "C:/Users/Administrator/Desktop/ML/DATA_Wave1_test_weight.xlsx"
    X_train, y_train = load_training_data(file_path_train)
    values = load_testing_data(file_path_test)

    epochs = 200
    learning_rate = 0.08
    input_nodes = 8
    hidden_nodes = 3
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8)
    
    results_boostrap_df, prob_BP = get_results(device, X_train, y_train, values, model, criterion, optimizer, scheduler, epochs)
    print(results_boostrap_df.mean())
    results_boostrap_df.to_excel("testing_weight_BP.xlsx", index=False)
    prob_BP.to_csv("testing-prob-weight-BP.csv", index=True)
        

# VIMP-cover
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path_train = "C:/Users/Administrator/Desktop/ML/DATA_Wave1_train_cover.xlsx"
    file_path_test = "C:/Users/Administrator/Desktop/ML/DATA_Wave1_test_cover.xlsx"
    X_train, y_train = load_training_data(file_path_train)
    values = load_testing_data(file_path_test)

    epochs = 200
    learning_rate = 0.1
    input_nodes = 4
    hidden_nodes = 2
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    results_boostrap_df, prob_BP = get_results(device, X_train, y_train, values, model, criterion, optimizer, scheduler, epochs)
    print(results_boostrap_df.mean())
    results_boostrap_df.to_excel("testing_cover_BP.xlsx", index=False)
    prob_BP.to_csv("testing-prob-cover-BP.csv", index=True)