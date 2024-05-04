# -*- coding: utf-8 -*-
"""


这部分是用"isotonic"校准预测概率并绘制校准曲线的代码
Isotonic  approach to calibrate probabilities and draw calibration curves
@author: Li Na and Hexiao Ding == Sun Yat-sen University
@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn


"""


import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# Set working path
work_path = 'C:/Users/86198/Desktop/ML-Ystich/ML/Stage4-exvalidation/exvali-prob'
os.chdir(work_path)


def Calibration_curve(df):
    # df is a DataFrame that contains the prediction probability and the true label
    true_labels = df["response"]  
    pred_probs = df["pred"]  
    # Create a logistic regression classifier
    logreg = LogisticRegression()

    # Training classifier
    logreg.fit(pred_probs.to_numpy().reshape(-1, 1), true_labels)
    
    # Calibration using a logistic regression classifier
    calibrated = CalibratedClassifierCV(logreg, method='isotonic', ensemble=False)
    calibrated.fit(pred_probs.to_numpy().reshape(-1, 1), true_labels)

    # Get the calibrated probability
    calibrated_probs = calibrated.predict_proba(pred_probs.to_numpy().reshape(-1, 1))[:, 1]

    # Calculate the Brier score
    brier_score = brier_score_loss(true_labels, calibrated_probs)

    # Calculated calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(true_labels, calibrated_probs, n_bins=10, strategy='uniform')

    # Draw calibration curves and histograms
    fig, ax1 = plt.subplots()

    # Plot the predicted probability distribution for all samples
    ax1.hist(calibrated_probs, range=(0, 1), bins=10, color='skyblue', alpha=0.5, label='All Predictions')

    # Plot the prediction probability distribution for the positive sample
    ax1.hist(calibrated_probs[true_labels == 1], range=(0, 1), bins=10, color='salmon', alpha=0.5, label='Positive Observations')

    ax1.set_ylabel('Sample size', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel('Predicted probability')
    ax1.set_ylim([0, ax1.get_ylim()[1]*1.2]) 

    # Draw a calibration curve on the secondary axis
    ax2 = ax1.twinx()
    ax2.plot(mean_predicted_value, fraction_of_positives, 's-', color='red', label='Calibration curve')
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax2.set_ylabel('Observed proportion', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([-0.05, 1.05])  # Y-axis of the curve

    # Add legend
    ax2.legend(loc="upper right", bbox_to_anchor=(0.43, 1.0), ncol=1)
    
    # Add title
    plt.title(f'Calibration Curve (Brier score: {brier_score:.3f})')
    plt.tight_layout()
    plt.show()
    return fig

# different datase
# origin
if __name__ == "__main__":
    data = pd.read_excel('vali_prob_origin_AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    fig = Calibration_curve(df)
    
if __name__ == "__main__":
    data = pd.read_excel('vali_prob_origin_AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    fig = Calibration_curve(df)

if __name__ == "__main__":
    df = pd.read_excel('vali_prob_origin_BP.xlsx')
    fig = Calibration_curve(df)

# SHAP
if __name__ == "__main__":
    data = pd.read_excel('vali_prob_SHAP_AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    fig = Calibration_curve(df)
    
if __name__ == "__main__":
    data = pd.read_excel('vali_prob_SHAP_AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    fig = Calibration_curve(df)

if __name__ == "__main__":
    df = pd.read_excel('vali_prob_SHAP_BP.xlsx')
    fig = Calibration_curve(df)
    
# lasso
if __name__ == "__main__":
    data = pd.read_excel('vali_prob_lasso_AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    fig = Calibration_curve(df)
    
if __name__ == "__main__":
    data = pd.read_excel('vali_prob_lasso_AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    fig = Calibration_curve(df)

if __name__ == "__main__":
    df = pd.read_excel('vali_prob_lasso_BP.xlsx')
    fig = Calibration_curve(df)

# gain
if __name__ == "__main__":
    data = pd.read_excel('vali_prob_gain_AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    fig = Calibration_curve(df)
    
if __name__ == "__main__":
    data = pd.read_excel('vali_prob_gain_AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    fig = Calibration_curve(df)

if __name__ == "__main__":
    df = pd.read_excel('vali_prob_gain_BP.xlsx')
    fig = Calibration_curve(df)

# weight
if __name__ == "__main__":
    data = pd.read_excel('vali_prob_weight_AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    fig = Calibration_curve(df)
    
if __name__ == "__main__":
    data = pd.read_excel('vali_prob_weight_AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    fig = Calibration_curve(df)

if __name__ == "__main__":
    df = pd.read_excel('vali_prob_weight_BP.xlsx')
    fig = Calibration_curve(df)

# cover
if __name__ == "__main__":
    data = pd.read_excel('vali_prob_cover_AdaLR.xlsx')
    df = data[data['Model'] == 'AdaBoost']
    fig = Calibration_curve(df)
    
if __name__ == "__main__":
    data = pd.read_excel('vali_prob_cover_AdaLR.xlsx')
    df = data[data['Model'] == 'Logistic Regression']
    fig = Calibration_curve(df)

if __name__ == "__main__":
    df = pd.read_excel('vali_prob_cover_BP.xlsx')
    fig = Calibration_curve(df)

