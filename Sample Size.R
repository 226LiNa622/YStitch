# 这部分是计算模型构建所需样本量的R代码
# Calculate the sample size required for model construction
# @author: Li Na and Hexiao Ding == Sun Yat-sen University
# @Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn

install.packages("pmsampsize")
library(pmsampsize)
c=0.90
pre=850/3038
# 25 featrures without feature selection
pmsampsize(type = "b",
           cstatistic = 0.90,
           parameters = 25, 
           prevalence = 850/3038)
n1 <- exp((-0.508+0.259*log(pre)+0.504*log(25)-log(0.050))/0.544)
n1
EPP1=n1*pre/25
EPP1

# 6 featrures by intersecting Top 10 features in XGBoost and RF model based on SHAP
pmsampsize(type = "b",
           cstatistic = 0.90,
           parameters = 6, 
           prevalence = 850/3038)
n2 <- exp((-0.508+0.259*log(pre)+0.504*log(6)-log(0.050))/0.544)
n2
EPP2=n2*pre/6
EPP2

# 22 featrures by LASSO method
pmsampsize(type = "b",
           cstatistic = 0.90,
           parameters = 22, 
           prevalence = 850/3038)
n3 <- exp((-0.508+0.259*log(pre)+0.504*log(22)-log(0.050))/0.544)
n3
EPP3=n3*pre/22
EPP3

# 5 featrures by intersecting Top 10 features in XGBoost and RF model based on VIMP-gain
pmsampsize(type = "b",
           cstatistic = 0.90,
           parameters = 5, 
           prevalence = 850/3038)
n4 <- exp((-0.508+0.259*log(pre)+0.504*log(5)-log(0.050))/0.544)
n4
EPP4=n4*pre/5
EPP4

# 8 featrures by intersecting Top 10 features in XGBoost and RF model based on VIMP-weight
pmsampsize(type = "b",
           cstatistic = 0.90,
           parameters = 8, 
           prevalence = 850/3038)
n5 <- exp((-0.508+0.259*log(pre)+0.504*log(8)-log(0.050))/0.544)
n5
EPP5=n5*pre/8
EPP5

# 4 featrures by intersecting Top 10 features in XGBoost and RF model based on VIMP-cover
pmsampsize(type = "b",
           cstatistic = 0.90,
           parameters = 4, 
           prevalence = 850/3038)
n6 <- exp((-0.508+0.259*log(pre)+0.504*log(4)-log(0.050))/0.544)
n6
EPP6=n6*pre/4
EPP6



