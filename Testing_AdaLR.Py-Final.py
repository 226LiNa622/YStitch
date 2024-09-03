# -*- coding: utf-8 -*-
"""

这部分是对测试集使用bootstrapping评价AdaBoost和LR训练模型的代码
Evaluate AdaBoost & LR model using bootstrapping on the test set
@author: Li Na and Hexiao Ding  ==  Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from lifelines.utils import concordance_index
from sklearn.utils import resample
from sklearn import ensemble
from sklearn.metrics import confusion_matrix

# Set working path
work_path = 'D:/Dataset'
os.chdir(work_path)

# Define data read method
# training data
def load_training_data(file_path_train):
    data = pd.read_excel(file_path_train)
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(data)
    new_data = pd.DataFrame(scaler.transform(data))
    x_train = new_data.iloc[:, :-1]
    y_train = new_data.iloc[:, -1]
    return x_train, y_train
# testing data
def load_testing_data(file_path_test):
    data = pd.read_excel(file_path_test)
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(data)
    test = pd.DataFrame(scaler.transform(data))
    return test

# define how to calculate sensitivity
def calculate_sensitivity(y_true, y_prob):
    y_pred = np.where(y_prob > 0.5, 1, 0)
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # calculate
    sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
    return sensitivity
# define how to calculate specificity
def calculate_specificity(y_true, y_prob):
    y_pred = np.where(y_prob > 0.5, 1, 0)
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # calculate
    specificity = cm[0][0] / (cm[0][0] + cm[0][1])
    return specificity

# Define the function of outputing model performance
def results(file_path_train, file_path_test, models):
    x_train, y_train = load_training_data(file_path_train)
    test = load_testing_data(file_path_test)
    values = test.values
    results_boostrap = []
    prob_list = []
    n_bootstrap = 1000  # Number of bootstrap iterations
    for i in range(n_bootstrap):
        subtest = resample(values, replace=True, n_samples=int(len(values)))
        x_test = subtest[:, :-1]
        y_test = subtest[:, -1]
        # Traverse different machine learning algorithms
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            # Prediction of the algorithm on the test data set
            y_predict = model.predict(x_test)
            # The probability of model predictions
            y_pred_proba = model.predict_proba(x_test)[:, 1]
            # Calculate various performance evaluation indicators
            acc = accuracy_score(y_test, y_predict)
            auc = roc_auc_score(y_test, y_pred_proba)
            f_score = f1_score(y_test, y_predict)
            c_index = concordance_index(y_test, y_pred_proba)
            sensitivity = calculate_sensitivity(y_test, y_pred_proba)
            specificity = calculate_specificity(y_test, y_pred_proba)
            results_boostrap.append({
                "n_bootstrap": i,
                "Model": model_name,
                "AUC": auc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ACC": acc,
                "F-score": f_score,
                "C-index": c_index
            })

            for pred, resp in zip(y_pred_proba, y_test):
                prob_list.append({
                    "n_bootstrap": i,
                    "Model": model_name,
                    "pred": pred,
                    "response": resp
                })
        # Convert the result list to a DataFrame
        results_boostrap_AdaLR = pd.DataFrame(results_boostrap)
        prob_AdaLR = pd.DataFrame(prob_list)
    return results_boostrap_AdaLR, prob_AdaLR
    
# Main function
# origin
if __name__ == "__main__":
    file_path_train = "DATA_Wave1_train.xlsx"
    file_path_test = "DATA_Wave1_test.xlsx"
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=116, learning_rate=0.5, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_boostrap_AdaLR, prob_AdaLR = results(file_path_train, file_path_test, models)
    results_boostrap_AdaLR.to_excel("test-origin-AdaLR.xlsx", index=False)
    prob_AdaLR.to_csv("test-prob-origin-AdaLR.csv", index=False) 
    
# SHAP
if __name__ == "__main__":
    file_path_train = "DATA_Wave1_train_SHAP.xlsx"
    file_path_test = "DATA_Wave1_test_SHAP.xlsx"
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=75, learning_rate=0.64, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_boostrap_AdaLR, prob_AdaLR = results(file_path_train, file_path_test, models)
    results_boostrap_AdaLR.to_excel("test-SHAP-AdaLR.xlsx", index=False)
    prob_AdaLR.to_csv("test-prob-SHAP-AdaLR.csv", index=False)

# lasso
if __name__ == "__main__":
    file_path_train = "DATA_Wave1_train_lasso.xlsx"
    file_path_test = "DATA_Wave1_test_lasso.xlsx"
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=113, learning_rate=0.76, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_boostrap_AdaLR, prob_AdaLR = results(file_path_train, file_path_test, models)
    results_boostrap_AdaLR.to_excel("test-lasso-AdaLR.xlsx", index=False)
    prob_AdaLR.to_csv("test-prob-lasso-AdaLR.csv", index=False) 
    
# gain
if __name__ == "__main__":
    file_path_train = "DATA_Wave1_train_gain.xlsx"
    file_path_test = "DATA_Wave1_test_gain.xlsx"
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=120, learning_rate=0.86, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_boostrap_AdaLR, prob_AdaLR = results(file_path_train, file_path_test, models)
    results_boostrap_AdaLR.to_excel("test-gain-AdaLR.xlsx", index=False)
    prob_AdaLR.to_csv("test-prob-gain-AdaLR.csv", index=False) 
    
# weight
if __name__ == "__main__":
    file_path_train = "DATA_Wave1_train_weight.xlsx"
    file_path_test = "DATA_Wave1_test_weight.xlsx"
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=0.39, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_boostrap_AdaLR, prob_AdaLR = results(file_path_train, file_path_test, models)
    results_boostrap_AdaLR.to_excel("test-weight-AdaLR.xlsx", index=False)
    prob_AdaLR.to_csv("test-prob-weight-AdaLR.csv", index=False) 
    
# cover
if __name__ == "__main__":
    file_path_train = "DATA_Wave1_train_cover.xlsx"
    file_path_test = "DATA_Wave1_test_cover.xlsx"
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=91, learning_rate=0.5800000000000001, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_boostrap_AdaLR, prob_AdaLR = results(file_path_train, file_path_test, models)
    results_boostrap_AdaLR.to_excel("test-cover-AdaLR.xlsx", index=False)
    prob_AdaLR.to_csv("test-prob-cover-AdaLR.csv", index=False) 
                                                                                                             
