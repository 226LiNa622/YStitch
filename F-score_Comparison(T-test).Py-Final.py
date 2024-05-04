# -*- coding: utf-8 -*-
"""


这部分是用使用两独立样本T检验对模型指标F-score进行两两比较的代码
The two independent sample T-test was used to compare the F-score of each two models
Isotonic  approach to calibrate probabilities and draw calibration curves
@author: Li Na and Hexiao Ding == Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""


import pandas as pd
import numpy as np
from scipy import stats
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
    df = pd.read_excel('c:/users/86198/desktop/ML-Ystich/ML/test_compare_all.xlsx')
    results = get_p(df)
    results.to_excel("c:/users/86198/desktop/ML-Ystich/ML/resutls_test_p.xlsx", index=True)
