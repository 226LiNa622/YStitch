# -*- coding: utf-8 -*-
"""


这部分是剪枝过程的代码
Feature selection process
@author: Li Na and Hexiao Ding == Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""


import os
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from sklearn.preprocessing import MinMaxScaler

# Set working path
work_path = 'D:/Dataset'
os.chdir(work_path)
# Read data set
DATA_Wave1 = pd.read_excel('RAW-DATA-Wave1.xlsx')
# Extract feature matrix
X = DATA_Wave1.iloc[:, 1:26]
# Extract dependent variable
y = DATA_Wave1.iloc[:, 26]


###############################################################################
###############################################################################
####################  切割(比例0.3)///  train:test==7:3  #######################


from sklearn.model_selection import train_test_split, StratifiedKFold

# train:test==7:3
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Export the training set and test set
DATA_Wave1_train = pd.concat([X_train, Y_train], axis=1)
DATA_Wave1_test = pd.concat([x_test, y_test], axis=1)
DATA_Wave1_train.to_excel("DATA_Wave1_train.xlsx", index=False)
DATA_Wave1_test.to_excel("DATA_Wave1_test.xlsx", index=False)

###############################################################################
###############################################################################
###########  数据的归一化///  Min-Max Scaler for data preprocessing  ############

def scal_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(data)
    new_data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    x = new_data.iloc[:, :-1]
    y = new_data.iloc[:, -1]
    return x, y
x_train, y_train = scal_data(DATA_Wave1_train)


# ###############################################################################
# ###############################################################################
# #####  剪枝对照 /// *Sensitivity Analysis==Purning [RF and XGBoost-VIMP]  ######


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance

features = x_train.columns.tolist()

# rf gini
rf_importance_list = []
rf = RandomForestClassifier(random_state=0, n_estimators=10000, n_jobs=-1)
rf.fit(x_train, y_train)
importances = rf.feature_importances_
for feature, importance in zip(features, importances):
    rf_importance_list.append({
        'Feature': feature,
        'Importance': importance})
rf_importance_list = pd.DataFrame(rf_importance_list)
# rf_importance_list.to_excel("rf_importance.xlsx", index=False)
print(rf_importance_list)

# xgb weight
xgb_importance_weight = []
xgb = XGBClassifier(random_state=0, n_estimators=10000, n_jobs=-1)
xgb.fit(x_train, y_train)
importances = xgb.feature_importances_
importances_weight = xgb.get_booster().get_score(importance_type='weight')
for feature, importance in zip(features, importances_weight.values()):
    xgb_importance_weight.append({
        'Feature': feature,
        'Importance': importance})
xgb_importance_weight = pd.DataFrame(xgb_importance_weight)
# xgb_importance_weight.to_excel("xgb_importance_weight.xlsx", index=False)
print(xgb_importance_weight)

# xgb gain
xgb_importance_gain = []
xgb = XGBClassifier(random_state=0, n_estimators=10000, n_jobs=-1)
xgb.fit(x_train, y_train)
importances = xgb.feature_importances_
importances_gain = xgb.get_booster().get_score(importance_type='gain')
for feature, importance in zip(features, importances_gain.values()):
    xgb_importance_gain.append({
        'Feature': feature,
        'Importance': importance})
xgb_importance_gain = pd.DataFrame(xgb_importance_gain)
# xgb_importance_gain.to_excel("xgb_importance_gain.xlsx", index=False)
print(xgb_importance_gain)

# xgb cover
xgb_importance_cover = []
xgb = XGBClassifier(random_state=0, n_estimators=10000, n_jobs=-1)
xgb.fit(x_train, y_train)
importances = xgb.feature_importances_
importances_cover = xgb.get_booster().get_score(importance_type='cover')
for feature, importance in zip(features, importances_cover.values()):
    xgb_importance_cover.append({
        'Feature': feature,
        'Importance': importance})
xgb_importance_cover = pd.DataFrame(xgb_importance_cover)
# xgb_importance_cover.to_excel("xgb_importance_cover.xlsx", index=False)
print(xgb_importance_cover)

xgb.get_booster().feature_names = features
plot_importance(xgb, importance_type='gain')  
pyplot.show()

# Intersection of Top 10 
# Based on weight: SLE.D1/SLE.D2/SLE.D4/SLE.D5/EI.D2/EI.D3/EI.D4/CS.D2
# Based on gain: SLE.D1/SLE.D3/EI.D2/EI.D3/SS.D3
# Based on cover: SLE.D1/SLE.D3/SLE.D5/SS.D3

# Generate feature filtered data sets for model construction
# Specify variables (columns) to keep
weight_keep = ['SLE.D1', 'SLE.D2', 'SLE.D4', 'SLE.D5','EI.D2', 'EI.D3', 'EI.D4', 'CS.D2', 'Outcome']
gain_keep = ['SLE.D1', 'SLE.D3', 'EI.D2', 'EI.D3', 'SS.D3', 'Outcome']
cover_keep = ['SLE.D1', 'SLE.D3', 'SLE.D5', 'SS.D3', 'Outcome']
# Create a new data box, keeping the specified variables
weight_df = DATA_Wave1_train[weight_keep]
gain_df = DATA_Wave1_train[gain_keep]
cover_df = DATA_Wave1_train[cover_keep]
weight_df.to_excel("Data_Wave1_train_weight.xlsx", index=False)
gain_df.to_excel("Data_Wave1_train_gain.xlsx", index=False)
cover_df.to_excel("Data_Wave1_train_cover.xlsx", index=False)


###############################################################################
###############################################################################
#######  剪枝对照 /// *Sensitivity Analysis==Purning [LASSO Regression]  #######


import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
import seaborn as sns
import numpy as np

# Define feature names
class Solution:
    def __init__(self, x_train, y_train):
        self.feature_names = ['Sex', 'Education Level', 'Social Role', 'City Size',
                              "Father's Occupation", "Mother's Occupation", 'MedStaffs in Family',
                              'Mental Consulting', 'Age', 'SS.D1', 'SS.D2', 'SS.D3', 'EI.D1', 'EI.D2',
                              'EI.D3', 'EI.D4', 'SLE.D1', 'SLE.D2', 'SLE.D3', 'SLE.D4', 'SLE.D5',
                              'CS.D1', 'CS.D2', 'CS.D3.SUB1', 'CS.D3.SUB2']
        self.x_train = x_train
        self.y_train = y_train
        self.lasso_best_alpha = self.optimal_lambda_value()

    # Get the best lambda value
    def optimal_lambda_value(self):
        Lambdas = np.logspace(-5, 2, 200)  
        lasso_cv = LassoCV(alphas=Lambdas, cv=10, max_iter=10000)
        lasso_cv.fit(self.x_train, y_train)
        
        # lambda.min
        lambda_min = lasso_cv.alpha_

        # lambda.1se
        mse_path = lasso_cv.mse_path_
        mean_mse = np.mean(mse_path, axis=1)
        std_mse = np.std(mse_path, axis=1) / np.sqrt(lasso_cv.cv)
        idx_min_mse = np.argmin(mean_mse)
        mse_1se = mean_mse[idx_min_mse] + std_mse[idx_min_mse]
        lambda_1se = lasso_cv.alphas_[mean_mse <= mse_1se][-1]

        print('lambda.min:', lambda_min)
        print('lambda.1se:', lambda_1se)

        lasso_coefficients = []

        # Store the partial regression coefficient of the model
        for Lambda in Lambdas:
            lasso = Lasso(alpha=Lambda, max_iter=10000)
            lasso.fit(self.x_train, self.y_train)
            lasso_coefficients.append(lasso.coef_)

        # Plot the relationship between Lambda and the regression lambda 1se coefficient
        plt.figure(figsize=(10, 6))
        plt.plot(Lambdas, lasso_coefficients)
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('Coefficients')
        #plt.title('LASSO Coefficients as a function of the regularization')
        plt.show()

        return lasso_cv.alpha_

    # Bring in the best lambda value training model
    def train_model(self):
        lasso = Lasso(alpha=self.lasso_best_alpha, max_iter=10000)
        lasso.fit(self.x_train, self.y_train)
        return lasso

    # Features and coefficient
    def feature_importance(self, lasso):
        # Create and export a DataFrame containing features and corresponding coefficients
        feature_names = self.x_train.columns.tolist()
        df = pd.DataFrame({'Feature': feature_names, 'Coeffcients': lasso.coef_})
        df.to_excel("lasso_coeffcients.xlsx", index=False)

        # Filter out the features whose coefficient is not 0
        nonzero_features = df[df['Coeffcients'] != 0]
        print("非零系数的特征和对应系数：\n", nonzero_features)

        # Plot
        plt.figure(figsize=(10, 10))
        nonzero_features = nonzero_features.sort_values(by='Coeffcients', ascending=True)
        sns.barplot(x="Coeffcients", y="Feature", data=nonzero_features)
        #plt.title('Lasso model')
        plt.tight_layout()
        plt.show()
        return nonzero_features


# Using
solution = Solution(x_train, y_train)
lasso_model = solution.train_model()
solution.feature_importance(lasso_model)

plt.show()

# Based on lasso: exclude EI.D4/SLE.D3/CS.D2

# Generate feature filtered data sets for model construction
# Specify variables (columns) to keep
lasso_exclude = ['EI.D4', 'SLE.D3', 'CS.D2']
# Create a new data box, keeping the specified variables
lasso_df = DATA_Wave1_train.drop(columns=lasso_exclude)
lasso_df.to_excel("Data_Wave1_train_lasso.xlsx", index=False)


###############################################################################
###############################################################################
###################  剪枝 /// *Purning [RF and XGBoost-SHAP]  #################


import shap
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


# ------------------------------剪枝(SHAP-XGBoost)-----------------------------

# Stores all SHAP values
shap_values_combined = []
np.random.seed(0)  # Set the random seed

# Make a list of random integers between 0 and 10000 with length CV_repeats as different data splits
n_repeats = 100 
random_states = np.random.randint(10000, size=n_repeats)
print(random_states)

# Add index column
x_train['index'] = range(len(x_train))

for random_state in random_states:
    
    # Store the SHAP value for each fold
    combined_df = pd.DataFrame()
    
    # Cross validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    for fold, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        
        # Divide the training set and test set
        x_train_fold, x_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Train XGBoost model
        model = XGBClassifier(random_state=0, n_estimators=10000, n_jobs=-1)
        model.fit(x_train_fold.drop(columns=['index']), y_train_fold)
        
        # SHAP to explain
        explainer = shap.TreeExplainer(model, approximate=False) # 'approximate=False' means that approximation is forbidden
        shap_values = explainer.shap_values(x_test_fold.drop(columns=['index']))
        
        # Adds the SHAP value to the index information
        shap_values_with_index = pd.DataFrame(shap_values, columns=x_test_fold.drop(columns=['index']).columns)
        shap_values_with_index['index'] = x_test_fold['index'].reset_index(drop=True)
        combined_df = pd.concat([combined_df, shap_values_with_index], ignore_index=True)
    
    # Reorder the SHAP values in the order of the samples in the original data
    sorted_shap_values = combined_df.sort_values(by='index')
    
    # Saves the SHAP values for each repeat to the overall list
    shap_values_combined.append(sorted_shap_values.drop(columns=['index']))

# Store SHAP values in the format (number of samples, number of features, number of repetitions)
shap_values_combined = np.swapaxes(shap_values_combined, 0, 1)
xgb_average_shap = np.mean(shap_values_combined, axis=1)

# Calculate the average SHAP value for each feature
mean_shap_values = np.mean(np.absolute(xgb_average_shap), axis=0)
mean_shap_dict = dict(zip(x_train.drop(columns=['index']).columns, mean_shap_values))

# Store SHAP values of each feature
xgb_shap=[]
for feature, mean_shap in mean_shap_dict.items():
    xgb_shap.append({
          'Feature': feature,
          'Mean SHAP Value': mean_shap})
xgb_shap_values = pd.DataFrame(xgb_shap)
# xgb_shap_values.to_excel("shap_xgb.xlsx", index=False)

# Visualization of the average SHAP value
shap.summary_plot(xgb_average_shap, features=x_train.drop(columns=['index']), feature_names=x_train.drop(columns=['index']).columns, show=False)
# plt.title("Average SHAP values after repeated cross-validation in XGBoost")
plt.show()

# Visualization of variation in SHAP values (range)
xgb_shap_range = np.max(shap_values_combined, axis=1) - np.min(shap_values_combined, axis=1)
# Convert data to a long format
xgb_range_long_df = pd.DataFrame(xgb_shap_range, columns=x_train.drop(columns=['index']).columns).melt(var_name='Features', value_name='Values')
# Data scaling
mean_abs_effects = xgb_range_long_df.groupby(['Features']).mean()
xgb_standardized = xgb_range_long_df.groupby(xgb_range_long_df.Features).transform(lambda x: x / x.mean())
xgb_standardized['Features'] = xgb_range_long_df.Features
# xgb_standardized.to_excel("shap_xgb_standardized.xlsx", index=False)
# Plot
# title = 'Range of SHAP values per feature across all cross-validation repeats in XGBoost'
# title = ' Scaled Range of SHAP values per feature across all cross-validation repeats in XGBoost'
xlab, ylab = 'Values', ''
# sns.catplot(data=xgb_range_long_df, x='Values', y='Features').set(xlabel=xlab, ylabel=ylab, title=title)
sns.catplot(data=xgb_standardized, x='Values', y='Features').set(xlabel=xlab, ylabel=ylab)
plt.tight_layout()
plt.show()



# ---------------------------------剪枝(SHAP-RF)-------------------------------

# Stores all SHAP values
shap_values_combined = []
np.random.seed(0)  # Set the random seed

# Make a list of random integers between 0 and 10000 with length CV_repeats as different data splits
n_repeats = 100 
random_states = np.random.randint(10000, size=n_repeats)
print(random_states)

# Add index column
x_train['index'] = range(len(x_train))

for random_state in random_states:
    
    # Store the SHAP value for each fold
    combined_df = pd.DataFrame()
    
    # Cross validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    for fold, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        
        # Divide the training set and test set
        x_train_fold, x_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Train XGBoost model
        model = RandomForestClassifier(random_state=0, n_estimators=1000, n_jobs=-1)
        model.fit(x_train_fold.drop(columns=['index']), y_train_fold)
        
        # SHAP to explain
        explainer = shap.TreeExplainer(model, approximate=False)# 'approximate=False' means that approximation is forbidden
        shap_values = explainer.shap_values(x_test_fold.drop(columns=['index']))[:, :, 1]
        
        # Adds the SHAP value to the index information
        shap_values_with_index = pd.DataFrame(shap_values, columns=x_test_fold.drop(columns=['index']).columns)
        shap_values_with_index['index'] = x_test_fold['index'].reset_index(drop=True)
        combined_df = pd.concat([combined_df, shap_values_with_index], ignore_index=True)
    
    # Reorder the SHAP values in the order of the samples in the original data
    sorted_shap_values = combined_df.sort_values(by='index')
    
    # Saves the SHAP values for each repeat to the overall list
    shap_values_combined.append(sorted_shap_values.drop(columns=['index']))

# Store SHAP values in the format (number of samples, number of features, number of repetitions)
shap_values_combined = np.swapaxes(shap_values_combined, 0, 1)
rf_average_shap = np.mean(shap_values_combined, axis=1)

# Calculate the average SHAP value for each feature
mean_shap_values = np.mean(np.absolute(rf_average_shap), axis=0)
mean_shap_dict = dict(zip(x_train.drop(columns=['index']).columns, mean_shap_values))

# Store SHAP values of each feature
rf_shap=[]
for feature, mean_shap in mean_shap_dict.items():
    rf_shap.append({
          'Feature': feature,
          'Mean SHAP Value': mean_shap})
rf_shap_values = pd.DataFrame(rf_shap)
# rf_shap_values.to_excel("shap_rf.xlsx", index=False)

# Visualization of the average SHAP value
shap.summary_plot(rf_average_shap, features=x_train.drop(columns=['index']), feature_names=x_train.drop(columns=['index']).columns, show=False)
#plt.title("Average SHAP values after repeated cross-validation in XGBoost")
plt.show()

# Visualization of variation in SHAP values (range)
rf_shap_range = np.max(shap_values_combined, axis=1) - np.min(shap_values_combined, axis=1)
# Convert data to a long format
rf_range_long_df = pd.DataFrame(rf_shap_range, columns=x_train.drop(columns=['index']).columns).melt(var_name='Features', value_name='Values')
# Data scaling
mean_abs_effects = rf_range_long_df.groupby(['Features']).mean()
rf_standardized = rf_range_long_df.groupby(rf_range_long_df.Features).transform(lambda x: x / x.mean())
rf_standardized['Features'] = rf_range_long_df.Features
# rf_standardized.to_excel("shap_rf_standardized.xlsx", index=False)
# Plot
# title = 'Range of SHAP values per feature across all cross-validation repeats in XGBoost'
# title = ' Scaled Range of SHAP values per feature across all cross-validation repeats in XGBoost'
xlab, ylab = 'Values', ''
# sns.catplot(data=xgb_range_long_df, x='Values', y='Features').set(xlabel=xlab, ylabel=ylab, title=title)
sns.catplot(data=rf_standardized, x='Values', y='Features').set(xlabel=xlab, ylabel=ylab)
plt.tight_layout()
plt.show()

# Intersection of Top 10
# SLE.D1/SLE.D4/SLE.D5/SS.D3/EI.D2/EI.D3

# Generate feature filtered data sets for model construction
# Specify variables (columns) to keep
shap_keep = ['SLE.D1', 'SLE.D4', 'SLE.D5','SS.D3', 'EI.D2', 'EI.D3', 'Outcome']
# Create a new data box, keeping the specified variables
shap_df = DATA_Wave1_train[shap_keep]
shap_df.to_excel("Data_Wave1_train_SHAP.xlsx", index=False)


