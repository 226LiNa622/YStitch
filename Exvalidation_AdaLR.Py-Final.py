# -*- coding: utf-8 -*-
"""

这部分是对预测模型AdaBoost和LR进行外部验证的代码
External validation of AdaBoost and LR model
@author: Li Na and Hexiao Ding  ==  Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""

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


##----------------------------Data pre-processing----------------------------##
##---------------------------------------------------------------------------##

# Set working path
work_path = 'D:/Dataset'
os.chdir(work_path)
# Read data set
DATA_Wave2 = pd.read_excel('RAW-DATA-Wave2（外部验证）.xlsx')
# Normalize-Wave2
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaler.fit(DATA_Wave2)
DATA_Wave2 = pd.DataFrame(scaler.transform(DATA_Wave2), columns=DATA_Wave2.columns)
DATA_Wave2.to_excel("DATA_Wave2.xlsx", index=False)

# Generate feature filtered data sets for model external validation
# weight
weight_keep = ['SLE.D1', 'SLE.D2', 'SLE.D4', 'SLE.D5','EI.D2', 'EI.D3', 'EI.D4', 'CS.D2', 'Outcome']
weight_df = DATA_Wave2[weight_keep]
weight_df.to_excel("DATA_Wave2_weight.xlsx", index=False)
# gain
gain_keep = ['SLE.D1', 'SLE.D3', 'EI.D2', 'EI.D3', 'SS.D3', 'Outcome']
gain_df = DATA_Wave2[gain_keep]
gain_df.to_excel("DATA_Wave2_gain.xlsx", index=False)
# cover
cover_keep = ['SLE.D1', 'SLE.D3', 'SLE.D5', 'SS.D3', 'Outcome']
cover_df = DATA_Wave2[cover_keep]
cover_df.to_excel("DATA_Wave2_cover.xlsx", index=False)
# SHAP
shap_keep = ['SLE.D1', 'SLE.D4', 'SLE.D5','SS.D3', 'EI.D2', 'EI.D3', 'Outcome']
shap_df = DATA_Wave2[shap_keep]
shap_df.to_excel("DATA_Wave2_SHAP.xlsx", index=False)
# lasso
lasso_exclude = ['EI.D4', 'SLE.D3', 'CS.D2']
lasso_df = DATA_Wave2.drop(columns=lasso_exclude)
lasso_df.to_excel("DATA_Wave2_lasso.xlsx", index=False)


##------------------------External validation: Ada&LR------------------------##
##---------------------------------------------------------------------------##

# Define data read
# training data
def load_training_data(file_path_train):
    data = pd.read_excel(file_path_train)
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(data)
    new_data = pd.DataFrame(scaler.transform(data))
    x_train = new_data.iloc[:, :-1]
    y_train = new_data.iloc[:, -1]
    return x_train, y_train
# validation data
def load_validation_data(file_path_val):
    data = pd.read_excel(file_path_val)
    x_validation = data.iloc[:, :-1]
    y_validation = data.iloc[:, -1]
    return x_validation, y_validation

# define how to get AdaBoost & LR model performance
def get_results(file_path_train, file_path_val, models):
    
    x_train, y_train = load_training_data(file_path_train)
    x_validation, y_validation = load_validation_data(file_path_val)
    results_list = []
    prob_list = []
    
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        # Prediction of the algorithm on externally validated data sets
        y_predict = model.predict(x_validation)
        # The probability of model predictions
        y_pred_proba = model.predict_proba(x_validation)[:, 1]
        # Calculate various performance evaluation indicators
        cm = confusion_matrix(y_validation, y_predict)
        acc = accuracy_score(y_validation, y_predict)
        auc = roc_auc_score(y_validation, y_pred_proba)
        f_score = f1_score(y_validation, y_predict)
        c_index = concordance_index(y_validation, y_pred_proba)
        sensitivity = sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        results_list.append({
            "Model": model_name,
            "AUC": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ACC": acc,
            "F-score": f_score,
            "C-index": c_index
        })
        for pred, resp in zip(y_pred_proba, y_validation):
            prob_list.append({
                "Model": model_name,
                "pred": pred,
                "response": resp
            })
    results_AdaLR = pd.DataFrame(results_list)
    prob_AdaLR = pd.DataFrame(prob_list)
    return results_AdaLR, prob_AdaLR

# origin
if __name__ == "__main__":
    file_path_train = "DATA_Wave1_train.xlsx"
    file_path_val = "DATA_Wave2.xlsx"
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=116, learning_rate=0.5, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_AdaLR, prob_AdaLR = get_results(file_path_train, file_path_val, models)
    results_AdaLR.to_excel("vali_origin_AdaLR.xlsx", index=False) 
    prob_AdaLR.to_excel("vali_prob_origin_AdaLR.xlsx", index=False) 
    
# SHAP
if __name__ == "__main__":
    file_path_train = "DATA_Wave1_train_SHAP.xlsx"
    file_path_val = "DATA_Wave2_SHAP.xlsx"
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=75, learning_rate=0.64, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_AdaLR, prob_AdaLR = get_results(file_path_train, file_path_val, models)
    results_AdaLR.to_excel("vali_SHAP_AdaLR.xlsx", index=False) 
    prob_AdaLR.to_excel("vali_prob_SHAP_AdaLR.xlsx", index=False) 

# lasso
if __name__ == "__main__":
    file_path_train = "DATA_Wave1_train_lasso.xlsx"
    file_path_val = "DATA_Wave2_lasso.xlsx"
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=113, learning_rate=0.76, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_AdaLR, prob_AdaLR = get_results(file_path_train, file_path_val, models)
    results_AdaLR.to_excel("vali_lasso_AdaLR.xlsx", index=False)
    prob_AdaLR.to_excel("vali_prob_lasso_AdaLR.xlsx", index=False) 
    
# gain
if __name__ == "__main__":
    file_path_train = "DATA_Wave1_train_gain.xlsx"
    file_path_val = "DATA_Wave2_gain.xlsx"
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=120, learning_rate=0.86, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_AdaLR, prob_AdaLR = get_results(file_path_train, file_path_val, models)
    results_AdaLR.to_excel("vali_gain_AdaLR.xlsx", index=False)
    prob_AdaLR.to_excel("vali_prob_gain_AdaLR.xlsx", index=False) 
    
# weight
if __name__ == "__main__":
    file_path_train = "DATA_Wave1_train_weight.xlsx"
    file_path_val = "DATA_Wave2_weight.xlsx"
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=0.39, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_AdaLR, prob_AdaLR = get_results(file_path_train, file_path_val, models)
    results_AdaLR.to_excel("vali_weight_AdaLR.xlsx", index=False)
    prob_AdaLR.to_excel("vali_prob_weight_AdaLR.xlsx", index=False)
    
# cover
if __name__ == "__main__":
    file_path_train = "DATA_Wave1_train_cover.xlsx"
    file_path_val = "DATA_Wave2_cover.xlsx"
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=91, learning_rate=0.5800000000000001, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_AdaLR, prob_AdaLR = get_results(file_path_train, file_path_val, models)
    results_AdaLR.to_excel("vali_cover_AdaLR.xlsx", index=False)
    prob_AdaLR.to_excel("vali_prob_cover_AdaLR.xlsx", index=False)
  
