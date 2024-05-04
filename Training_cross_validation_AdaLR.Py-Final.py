# -*- coding: utf-8 -*-
"""

这部分是对AdaBoost和LR预测模型在训练集进行10折交叉验证的代码
10-fold cross validation of AdaBoost and LR prediction models in the training set
@author: Li Na and Hexiao Ding  ==  Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import ensemble
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import pandas as pd
import warnings
from lifelines.utils import concordance_index
warnings.filterwarnings("ignore")
import os
from sklearn.preprocessing import MinMaxScaler

# Set working path
work_path = 'C:/Users/86198/Desktop/ML'
os.chdir(work_path)

def load_data(file_path):
    data = pd.read_excel(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(data)
    new_data = pd.DataFrame(scaler.transform(data))
    x = new_data.iloc[:, :-1]
    y = new_data.iloc[:, -1]
    return x, y

def calculate_sensitivity(y_true, y_prob):
    y_pred = np.where(y_prob > 0.5, 1, 0)
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # calculate
    sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
    return sensitivity

def calculate_specificity(y_true, y_prob):
    y_pred = np.where(y_prob > 0.5, 1, 0)
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # calculate
    specificity = cm[0][0] / (cm[0][0] + cm[0][1])
    return specificity

# Define a function that calculates models' performance
def results(x, y, models):
    # Dataframe for storage
    global results_AdaLR, prob_AdaLR
    results_list = []
    prob_list = []
    # Initialize RepeatedStratifiedKFold
    kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=0)
    # models = the AdaBoost & LR algorithm with optimum parameter
    # Iterate through different folds and different repeats
    for fold, (train_idx, test_idx) in enumerate(kf.split(x, y)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Algorithm fitting on training data set
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            # Prediction of the algorithm on the testing data set
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
            results_list.append({
                "Model": model_name,
                "Fold": fold,
                "AUC": auc,
                "ACC": acc,
                "F-score": f_score,
                "C-index": c_index,
                "sensitivity": sensitivity,
                "specificity": specificity
            })
            for pred, resp in zip(y_pred_proba, y_test):
                prob_list.append({
                    "fold": fold,
                    "Model": model_name,
                    "pred": pred,
                    "response": resp
                })
        results_AdaLR = pd.DataFrame(results_list)
        prob_AdaLR = pd.DataFrame(prob_list)
    return results_AdaLR, prob_AdaLR


# origin
if __name__ == "__main__":
    file_path = "DATA_Wave1_train.xlsx"
    x, y = load_data(file_path)
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=116, learning_rate=0.5, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_AdaLR, prob_AdaLR = results(x, y, models)
    results_AdaLR.to_excel("kfold_origin_AdaLR.xlsx", index=False)
    prob_AdaLR.to_csv("kfold_prob_origin_AdaLR.csv", index=False)

# SHAP
if __name__ == "__main__":
    file_path = "DATA_Wave1_train_SHAP.xlsx"
    x, y = load_data(file_path)
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=75, learning_rate=0.64, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_AdaLR, prob_AdaLR = results(x, y, models)
    results_AdaLR.to_excel("kfold_SHAP_AdaLR.xlsx", index=False)
    prob_AdaLR.to_csv("kfold_prob_SHAP_AdaLR.csv", index=False)

# lasso
if __name__ == "__main__":
    file_path = "DATA_Wave1_train_lasso.xlsx"
    x, y = load_data(file_path)
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=113, learning_rate=0.76, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_AdaLR, prob_AdaLR = results(x, y, models)
    results_AdaLR.to_excel("kfold_lasso_AdaLR.xlsx", index=False)
    prob_AdaLR.to_csv("kfold_prob_lasso_AdaLR.csv", index=False)

# gain
if __name__ == "__main__":
    file_path = "DATA_Wave1_train_gain.xlsx"
    x, y = load_data(file_path)
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=120, learning_rate=0.86, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_AdaLR, prob_AdaLR = results(x, y, models)
    results_AdaLR.to_excel("kfold_gain_AdaLR.xlsx", index=False)
    prob_AdaLR.to_csv("kfold_prob_gain_AdaLR.csv", index=False)

# weight
if __name__ == "__main__":
    file_path = "DATA_Wave1_train_weight.xlsx"
    x, y = load_data(file_path)
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=0.39, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_AdaLR, prob_AdaLR = results(x, y, models)
    results_AdaLR.to_excel("kfold_weight_AdaLR.xlsx", index=False)
    prob_AdaLR.to_csv("kfold_prob_weight_AdaLR.csv", index=False)

# cover
if __name__ == "__main__":
    file_path = "DATA_Wave1_train_cover.xlsx"
    x, y = load_data(file_path)
    models = {
        "AdaBoost": ensemble.AdaBoostClassifier(n_estimators=91, learning_rate=0.5800000000000001, random_state=0),
        "Logistic Regression": LogisticRegression(n_jobs=-1, random_state=0)
    }
    results_AdaLR, prob_AdaLR = results(x, y, models)
    results_AdaLR.to_excel("kfold_cover_AdaLR.xlsx", index=False)
    prob_AdaLR.to_csv("kfold_prob_cover_AdaLR.csv", index=False)

