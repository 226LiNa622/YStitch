# 这部分是绘制临床增益曲线的R代码
# draw clinical impact curves
# @author: Li Na and Hexiao Ding == Sun Yat-sen University
# @Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn

library(readxl) 
library(openxlsx)
library(rms)
library(rmda)
library(glue)
library(ggplot2)
setwd("C:/Users/86198/Desktop/ML-Ystich/ML/Stage4-exvalidation/exvali-prob")

# origin+LR
data<-read_excel("vali_prob_origin_AdaLR.xlsx")
df <- data[data$Model == "Logistic Regression", ]
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "origin_LR.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# origin+AdaBoost
data<-read_excel("vali_prob_origin_AdaLR.xlsx")
df <- data[data$Model == "AdaBoost", ]
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "origin_AdaBoost.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# origin+BPNN
df<-read_excel("vali_prob_origin_BP.xlsx")
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "origin_BPNN.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# SHAP+LR
data<-read_excel("vali_prob_SHAP_AdaLR.xlsx")
df <- data[data$Model == "Logistic Regression", ]
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "SHAP_LR.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# SHAP+AdaBoost
data<-read_excel("vali_prob_SHAP_AdaLR.xlsx")
df <- data[data$Model == "AdaBoost", ]
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "SHAP_AdaBoost.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# SHAP+BPNN
df<-read_excel("vali_prob_SHAP_BP.xlsx")
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "SHAP_BPNN.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# lasso+LR
data<-read_excel("vali_prob_lasso_AdaLR.xlsx")
df <- data[data$Model == "Logistic Regression", ]
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "lasso_LR.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# lasso+AdaBoost
data<-read_excel("vali_prob_lasso_AdaLR.xlsx")
df <- data[data$Model == "AdaBoost", ]
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "lasso_AdaBoost.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

#lasso+BPNN
df<-read_excel("vali_prob_lasso_BP.xlsx")
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "lasso_BPNN.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# gain+LR
data<-read_excel("vali_prob_gain_AdaLR.xlsx")
df <- data[data$Model == "Logistic Regression", ]
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "gain_LR.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# gain+AdaBoost
data<-read_excel("vali_prob_gain_AdaLR.xlsx")
df <- data[data$Model == "AdaBoost", ]
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "gain_AdaBoost.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

#gain+BPNN
df<-read_excel("vali_prob_gain_BP.xlsx")
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "gain_BPNN.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# weight+LR
data<-read_excel("vali_prob_weight_AdaLR.xlsx")
df <- data[data$Model == "Logistic Regression", ]
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "weight_LR.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# weight+AdaBoost
data<-read_excel("vali_prob_weight_AdaLR.xlsx")
df <- data[data$Model == "AdaBoost", ]
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "weight_AdaBoost.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

#weight+BPNN
df<-read_excel("vali_prob_weight_BP.xlsx")
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "weight_BPNN.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# cover+LR
data<-read_excel("vali_prob_cover_AdaLR.xlsx")
df <- data[data$Model == "Logistic Regression", ]
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "cover_LR.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

# cover+AdaBoost
data<-read_excel("vali_prob_cover_AdaLR.xlsx")
df <- data[data$Model == "AdaBoost", ]
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "cover_AdaBoost.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()

#cover+BPNN
df<-read_excel("vali_prob_cover_BP.xlsx")
# Calculate the normalized net income using decision curve analysis
dc_result <- rmda::decision_curve(response ~ pred,
                                  data = df,
                                  family = binomial(),
                                  thresholds = seq(0, 1, by = 0.01))
# Draw decision curves using rmda packages
rmda::plot_decision_curve(dc_result, col = "#E64B35B2", standardize = FALSE)

# Clinical impact curve was drawn
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
# Sets the size and resolution of the saved image
jpeg(filename = "cover_BPNN.jpg", width = 8*400, height = 6*400, res = 400)
plot_clinical_impact(dc_result,
                     population.size = 1000, # Hypothetical crowd size
                     n.cost.benefits = 8, # Quantity based on cost-benefit ratio
                     confidence.intervals = FALSE, 
                     col=c('red','blue'),
                     lty = 1,
                     lwd = 2,
                     cost.benefit.xlab = "Cost:Benefit Ratio",
                     legend.position = "topright")
dev.off()