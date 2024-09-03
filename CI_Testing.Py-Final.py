# -*- coding: utf-8 -*-
"""

这部分是计算在bootstrapping生成的测试子集上模型性能指标95%CI的代码
Calculate 95%CI of the model performance index on the test subset after bootstrapping
@author: Li Na and Hexiao Ding  ==  Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""
from scipy import stats
import os
import pandas as pd

# Set working path
work_path = 'D:/ML-Ystich/YStitch/Dataset'
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
    data = pd.read_excel('test-origin-AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-origin-LR.xlsx", index=True)
    print(result_df)
if __name__ == "__main__":
    data = pd.read_excel('test-origin-AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-origin-Ada.xlsx", index=True)
    print(result_df)
if __name__ == "__main__":
    df = pd.read_excel('test-origin-BP.xlsx')
    selected_columns = df.iloc[:, 1:7]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-origin-BP.xlsx", index=True)
    print(result_df)
    
# SHAP
if __name__ == "__main__":
    data = pd.read_excel('test-SHAP-AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-SHAP-LR.xlsx", index=True)
    print(result_df)
if __name__ == "__main__":
    data = pd.read_excel('test-SHAP-AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-SHAP-Ada.xlsx", index=True)
    print(result_df)
if __name__ == "__main__":
    df = pd.read_excel('test-SHAP-BP.xlsx')
    selected_columns = df.iloc[:, 1:7]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-SHAP-BP.xlsx", index=True)
    print(result_df)
    
# LASSO
if __name__ == "__main__":
    data = pd.read_excel('test-lasso-AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-lasso-LR.xlsx", index=True)
    print(result_df)
if __name__ == "__main__":
    data = pd.read_excel('test-lasso-AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-lasso-Ada.xlsx", index=True)
    print(result_df)
if __name__ == "__main__":
    df = pd.read_excel('test-lasso-BP.xlsx')
    selected_columns = df.iloc[:, 1:7]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-lasso-BP.xlsx", index=True)
    print(result_df)
    
# gain
if __name__ == "__main__":
    data = pd.read_excel('test-gain-AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-gain-LR.xlsx", index=True)
    print(result_df)
if __name__ == "__main__":
    data = pd.read_excel('test-gain-AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-gain-Ada.xlsx", index=True)
    print(result_df)
if __name__ == "__main__":
    df = pd.read_excel('test-gain-BP.xlsx')
    selected_columns = df.iloc[:, 1:7]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-gain-BP.xlsx", index=True)
    print(result_df)
    
# weight
if __name__ == "__main__":
    data = pd.read_excel('test-weight-AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-weight-LR.xlsx", index=True)
    print(result_df)
if __name__ == "__main__":
    data = pd.read_excel('test-weight-AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-weight-Ada.xlsx", index=True)
    print(result_df)
if __name__ == "__main__":
    df = pd.read_excel('test-weight-BP.xlsx')
    selected_columns = df.iloc[:, 1:7]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-weight-BP.xlsx", index=True)
    print(result_df)
    
# cover
if __name__ == "__main__":
    data = pd.read_excel('test-cover-AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-cover-LR.xlsx", index=True)
    print(result_df)
if __name__ == "__main__":
    data = pd.read_excel('test-cover-AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    selected_columns = df.iloc[:, 2:8]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-cover-Ada.xlsx", index=True)
    print(result_df)
if __name__ == "__main__":
    df = pd.read_excel('test-cover-BP.xlsx')
    selected_columns = df.iloc[:, 1:7]
    result_df = calculating_CI(selected_columns)
    result_df.to_excel("CI-test-cover-BP.xlsx", index=True)
    print(result_df)
    
