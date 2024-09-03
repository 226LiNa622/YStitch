# -*- coding: utf-8 -*-
"""

这部分是计算训练集模型交叉验证性能指标95%CI的代码
Compute 95% CI of model cross-validation performance in the training set
@author: Li Na and Hexiao Ding  ==  Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""
from scipy import stats
import os
import pandas as pd

# Set working path
work_path = 'D:/dataset'
os.chdir(work_path)

def calculating_CI(selected_columns):
    # Calculate the mean and standard errors
    mean = selected_columns.mean()
    std_error = selected_columns.sem()

    # Calculate 95% confidence intervals
    confidence_interval = stats.t.interval(0.95, len(selected_columns)-1, loc=mean, scale=std_error)

    # Create a DataFrame and set the index
    result_df = pd.DataFrame({'Mean': mean,
                          'Standard Error': std_error,
                          'Confidence Interval Lower': confidence_interval[0],
                          'Confidence Interval Upper': confidence_interval[1]},
                         index=selected_columns.columns)
    return result_df

# origin
if __name__ == "__main__":
    data = pd.read_excel('Kfold_origin_AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-origin-LR.xlsx", index=True)
if __name__ == "__main__":
    data = pd.read_excel('Kfold_origin_AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-origin-Ada.xlsx", index=True)
if __name__ == "__main__":
    df = pd.read_excel('Kfold_origin_BP.xlsx')
    selected_columns = df.iloc[:, 1:7]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-origin-BP.xlsx", index=True)
    
# SHAP
if __name__ == "__main__":
    data = pd.read_excel('Kfold_SHAP_AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-SHAP-LR.xlsx", index=True)
if __name__ == "__main__":
    data = pd.read_excel('Kfold_SHAP_AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-SHAP-Ada.xlsx", index=True)
if __name__ == "__main__":
    df = pd.read_excel('Kfold_SHAP_BP.xlsx')
    selected_columns = df.iloc[:, 1:7]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-SHAP-BP.xlsx", index=True)
    
# LASSO
if __name__ == "__main__":
    data = pd.read_excel('Kfold_lasso_AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-lasso-LR.xlsx", index=True)
if __name__ == "__main__":
    data = pd.read_excel('Kfold_lasso_AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-lasso-Ada.xlsx", index=True)
if __name__ == "__main__":
    df = pd.read_excel('Kfold_lasso_BP.xlsx')
    selected_columns = df.iloc[:, 1:7]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-lasso-BP.xlsx", index=True)
    
# gain
if __name__ == "__main__":
    data = pd.read_excel('Kfold_gain_AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-gain-LR.xlsx", index=True)
if __name__ == "__main__":
    data = pd.read_excel('Kfold_gain_AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-gain-Ada.xlsx", index=True)
if __name__ == "__main__":
    df = pd.read_excel('Kfold_gain_BP.xlsx')
    selected_columns = df.iloc[:, 1:7]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-gain-BP.xlsx", index=True)
    
# weight
if __name__ == "__main__":
    data = pd.read_excel('Kfold_weight_AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-weight-LR.xlsx", index=True)
if __name__ == "__main__":
    data = pd.read_excel('Kfold_weight_AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-weight-Ada.xlsx", index=True)
if __name__ == "__main__":
    df = pd.read_excel('Kfold_weight_BP.xlsx')
    selected_columns = df.iloc[:, 1:7]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-weight-BP.xlsx", index=True)
    
# cover
if __name__ == "__main__":
    data = pd.read_excel('Kfold_cover_AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-cover-LR.xlsx", index=True)
if __name__ == "__main__":
    data = pd.read_excel('Kfold_cover_AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-cover-Ada.xlsx", index=True)
if __name__ == "__main__":
    df = pd.read_excel('Kfold_cover_BP.xlsx')
    selected_columns = df.iloc[:, 1:7]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-kfold-cover-BP.xlsx", index=True)



