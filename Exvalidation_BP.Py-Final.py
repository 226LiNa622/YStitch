# -*- coding: utf-8 -*-
"""

这部分是对BPNN模型进行外部验证的代码
External validation of BPNN model
@author: Li Na and Hexiao Ding  ==  Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import warnings
from lifelines.utils import concordance_index
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Set working path
work_path = 'D:/Dataset'
os.chdir(work_path)


##-------------------------External validation: BPNN-------------------------##
##---------------------------------------------------------------------------##

class BPNet(nn.Module):
    def __init__(self, input_num, hide_num, output_num):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_num, hide_num)
        self.fc2 = nn.Linear(hide_num, output_num)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Define data read
# training data
def load_training_data(file_path_train):
    data = pd.read_excel(file_path_train)
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(data)
    new_data = pd.DataFrame(scaler.transform(data))
    x_train = new_data.iloc[:, :-1].values.astype(np.float32)
    y_train = new_data.iloc[:, -1].values.astype(np.float32)
    return x_train, y_train
# validation data
def load_validation_data(file_path_val):
    data = pd.read_excel(file_path_val)
    x_validation = data.iloc[:, :-1].values.astype(np.float32)
    y_validation = data.iloc[:, -1].values.astype(np.float32)
    return x_validation, y_validation

# define how to get BP model performance
def get_results(device, file_path_train, file_path_val, model, initial_state, criterion, optimizer, scheduler):
    x_train, y_train = load_training_data(file_path_train)
    x_validation, y_validation = load_validation_data(file_path_val)
    results_list = []
    prob_list = []
    
    model.load_state_dict(initial_state)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    X_train_tensor = torch.FloatTensor(x_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(x_validation).to(device)
    y_val_tensor = torch.FloatTensor(y_validation).to(device)

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
            
        pass

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
        
    results_list.append({
            "AUC": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ACC": acc,
            "F-SCORE": f_score,
            "C-index": c_index,
    })
    
    for pred, resp in zip(val_outputs.cpu().numpy(), y_val_tensor.cpu().numpy()):
        prob_list.append({
            "pred": pred,
            "response": resp
        })
    
    # # calibration curve
    # prob_true, prob_pred = calibration_curve(y_val_tensor.cpu().numpy(), val_outputs.cpu().numpy(), n_bins=10, strategy='uniform')
    # plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    # plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    # plt.xlabel('Predicted probability')
    # plt.ylabel('True probability in each bin')
    # plt.legend()
    # plt.show()
    
    # 将结果列表转为DataFrame
    results_BP = pd.DataFrame(results_list)
    prob_BP = pd.DataFrame(prob_list)
    return results_BP, prob_BP

# origin
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path_train = "DATA_Wave1_train.xlsx"
    file_path_val = "DATA_Wave2.xlsx"
    
    epochs = 250
    learning_rate = 0.1
    input_nodes = 25
    hidden_nodes = 6
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes).to(device)
    initial_state = {name: tensor.to(device) for name, tensor in model.state_dict().items()}
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    results_BP, prob_BP = get_results(device, file_path_train, file_path_val, model, initial_state, criterion, optimizer, scheduler)
    results_BP.to_excel("vali_origin_BP.xlsx", index=False) 
    prob_BP.to_excel("vali_prob_origin_BP.xlsx", index=False) 

# SHAP
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path_train = "DATA_Wave1_train_SHAP.xlsx"
    file_path_val = "DATA_Wave2_SHAP.xlsx"
    
    epochs = 200
    learning_rate = 0.1
    input_nodes = 6
    hidden_nodes = 3
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes).to(device)
    initial_state = {name: tensor.to(device) for name, tensor in model.state_dict().items()}
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    results_BP, prob_BP = get_results(device, file_path_train, file_path_val, model, initial_state, criterion, optimizer, scheduler)
    results_BP.to_excel("vali_SHAP_BP.xlsx", index=False) 
    prob_BP.to_excel("vali_prob_SHAP_BP.xlsx", index=False) 

# lasso
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path_train = "DATA_Wave1_train_lasso.xlsx"
    file_path_val = "DATA_Wave2_lasso.xlsx"
    
    epochs = 200
    learning_rate = 0.1
    input_nodes = 22
    hidden_nodes = 6
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes).to(device)
    initial_state = {name: tensor.to(device) for name, tensor in model.state_dict().items()}
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    results_BP, prob_BP = get_results(device, file_path_train, file_path_val, model, initial_state, criterion, optimizer, scheduler)
    results_BP.to_excel("vali_lasso_BP.xlsx", index=False) 
    prob_BP.to_excel("vali_prob_lasso_BP.xlsx", index=False) 
    
# gain
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path_train = "DATA_Wave1_train_gain.xlsx"
    file_path_val = "DATA_Wave2_gain.xlsx"
    
    epochs = 200
    learning_rate = 0.1
    input_nodes = 5
    hidden_nodes = 2
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes).to(device)
    initial_state = {name: tensor.to(device) for name, tensor in model.state_dict().items()}
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    results_BP, prob_BP = get_results(device, file_path_train, file_path_val, model, initial_state, criterion, optimizer, scheduler)
    results_BP.to_excel("vali_gain_BP.xlsx", index=False) 
    prob_BP.to_excel("vali_prob_gain_BP.xlsx", index=False) 
    
# weight
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path_train = "DATA_Wave1_train_weight.xlsx"
    file_path_val = "DATA_Wave2_weight.xlsx"
   
    epochs = 200
    learning_rate = 0.08
    input_nodes = 8
    hidden_nodes = 3
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes).to(device)
    initial_state = {name: tensor.to(device) for name, tensor in model.state_dict().items()}
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8)
    
    results_BP, prob_BP = get_results(device, file_path_train, file_path_val, model, initial_state, criterion, optimizer, scheduler)
    results_BP.to_excel("vali_weight_BP.xlsx", index=False) 
    prob_BP.to_excel("vali_prob_weight_BP.xlsx", index=False) 
    
# cover
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_path_train = "DATA_Wave1_train_cover.xlsx"
    file_path_val = "DATA_Wave2_cover.xlsx"
   
    epochs = 200
    learning_rate = 0.1
    input_nodes = 4
    hidden_nodes = 2
    output_nodes = 1

    model = BPNet(input_nodes, hidden_nodes, output_nodes).to(device)
    initial_state = {name: tensor.to(device) for name, tensor in model.state_dict().items()}
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    results_BP, prob_BP = get_results(device, file_path_train, file_path_val, model, initial_state, criterion, optimizer, scheduler)
    results_BP.to_excel("vali_cover_BP.xlsx", index=False) 
    prob_BP.to_excel("vali_prob_cover_BP.xlsx", index=False) 
