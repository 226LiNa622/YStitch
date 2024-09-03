# -*- coding: utf-8 -*-
"""


这部分是用使用两独立样本T检验对模型指标F-score进行两两比较的代码
The two independent sample T-test was used to compare the F-score of each two models
@author: Li Na and Hexiao Ding == Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""


import pandas as pd
import numpy as np
from scipy import stats

# Set the working directory
path = "D:/Dataset"

# Read Excel files and add the Dataset_Type column
df1 = pd.read_excel(f"{path}/test-origin-BP.xlsx")
df2 = pd.read_excel(f"{path}/test-SHAP-BP.xlsx")
df3 = pd.read_excel(f"{path}/test-lasso-BP.xlsx")
df4 = pd.read_excel(f"{path}/test-gain-BP.xlsx")
df5 = pd.read_excel(f"{path}/test-weight-BP.xlsx")
df6 = pd.read_excel(f"{path}/test-cover-BP.xlsx")

df1['Dataset_Type'] = "ORIGIN"
df2['Dataset_Type'] = "SHAP"
df3['Dataset_Type'] = "LASSO"
df4['Dataset_Type'] = "GAIN"
df5['Dataset_Type'] = "WEIGHT"
df6['Dataset_Type'] = "COVER"

# Combine the BP datasets
data1 = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
data1['Algorithm_Type'] = "BPNN"

# Read and process the AdaLR datasets
df7 = pd.read_excel(f"{path}/test-origin-AdaLR.xlsx")
df8 = pd.read_excel(f"{path}/test-SHAP-AdaLR.xlsx")
df9 = pd.read_excel(f"{path}/test-lasso-AdaLR.xlsx")
df10 = pd.read_excel(f"{path}/test-gain-AdaLR.xlsx")
df11 = pd.read_excel(f"{path}/test-weight-AdaLR.xlsx")
df12 = pd.read_excel(f"{path}/test-cover-AdaLR.xlsx")

df7['Dataset_Type'] = "ORIGIN"
df8['Dataset_Type'] = "SHAP"
df9['Dataset_Type'] = "LASSO"
df10['Dataset_Type'] = "GAIN"
df11['Dataset_Type'] = "WEIGHT"
df12['Dataset_Type'] = "COVER"

# Combine the AdaLR datasets
data2 = pd.concat([df7, df8, df9, df10, df11, df12], ignore_index=True)

# Rename the "Model" column to "Algorithm_Type"
data2.rename(columns={'Model': 'Algorithm_Type'}, inplace=True)

# Replace "Logistic Regression" with "LR"
data2['Algorithm_Type'] = data2['Algorithm_Type'].replace("Logistic Regression", "LR")

# Select and combine the relevant columns
data1 = data1[['n_bootstrap', 'Dataset_Type', 'Algorithm_Type', 'F-SCORE']]
data2 = data2[['n_bootstrap', 'Dataset_Type', 'Algorithm_Type', 'F-score']]

# Standardize the column name
data2.rename(columns={'F-score': 'F-SCORE'}, inplace=True)
data = pd.concat([data1, data2], ignore_index=True)

# View the combined data
print(data.head())

# Export the combined data to an Excel file
data.to_excel(f"{path}/test_compare_all.xlsx", index=False)


def get_p(df):
    # Generate all possible combinations of model types
    all_combinations = [(algo, ds) for algo in df['Dataset_Type'].unique() for ds in df['Algorithm_Type'].unique()]

    # Initialization of the resulting data frame
    results_df = []

    # Compare pairwise model types
    for i in range(len(all_combinations)):
        for j in range(i+1, len(all_combinations)):
            model_1, ds_1 = all_combinations[i]
            model_2, ds_2 = all_combinations[j]
                
            # Extract F-score values for both model types
            Fscore_1 = df[(df['Dataset_Type'] == model_1) & (df['Algorithm_Type'] == ds_1)]['F-score']
            Fscore_2 = df[(df['Dataset_Type'] == model_2) & (df['Algorithm_Type'] == ds_2)]['F-score']

            if len(Fscore_1) == 0 or len(Fscore_2) == 0:
                continue  # Skip combinations without data
                
            # Two independent samples t test was performed
            t_stat, p_value = stats.ttest_ind(Fscore_1, Fscore_2, equal_var=False)
                
            # Calculate Difference, Standard Error, and 95% confidence intervals
            Fscore_diff = np.mean(Fscore_1) - np.mean(Fscore_2)
            se = np.sqrt(np.var(Fscore_1)/len(Fscore_1) + np.var(Fscore_2)/len(Fscore_2))
            ci_lower, ci_upper = stats.t.interval(0.95, len(Fscore_1)-1, loc=Fscore_diff, scale=se)
                
            # Add a result to the result data box
            comparison_model = f"{model_1}-{ds_1} vs. {model_2}-{ds_2}"
            results_df.append({'Comparison_Models': comparison_model, 
                               'F-score Difference': Fscore_diff,
                               'Standard Error': se, 
                               '95% CI Lower': ci_lower, 
                               '95% CI Upper': ci_upper,
                               'P-value': p_value})

    results = pd.DataFrame(results_df)
    return results

if __name__ == "__main__":
    df = pd.read_excel(f"{path}/test_compare_all.xlsx")
    results = get_p(df)
    results.to_excel(f"{path}/resutls_test_p.xlsx", index=True)
