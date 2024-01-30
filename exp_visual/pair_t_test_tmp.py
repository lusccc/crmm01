import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
data1 = {
    'Model': ['RF', 'ET', 'XGBoost', 'LogR', 'SVM', 'MLP', 'KNN', 'GBDT', 'Bagging', 'AdaBoost', 'MEDBN (ours)'],
    'Acc': [0.8310, 0.7895, 0.8560, 0.8172, 0.7313, 0.7202, 0.6233, 0.8449, 0.7729, 0.8061, 0.8781],
    'AUC': [0.9082, 0.8828, 0.9124, 0.8730, 0.7827, 0.7794, 0.6331, 0.9078, 0.8740, 0.8824, 0.9391],
    'KS': [0.6928, 0.6257, 0.7135, 0.6545, 0.4508, 0.4241, 0.2088, 0.6970, 0.5645, 0.6425, 0.7607],
    'G-mean': [0.8240, 0.7606, 0.8491, 0.8033, 0.7132, 0.6935, 0.5676, 0.8355, 0.7746, 0.7931, 0.8673]
}

data2 = {
    'Model': ['RF', 'ET', 'XGBoost', 'LogR', 'SVM', 'MLP', 'KNN', 'GBDT', 'Bagging', 'AdaBoost', 'MEDBN (ours)'],
    'Acc': [0.9386, 0.9289, 0.9431, 0.9237, 0.9034, 0.9132, 0.8413, 0.8840, 0.9229, 0.8331, 0.9693],
    'AUC': [0.9808, 0.9768, 0.9815, 0.9717, 0.9546, 0.9552, 0.9039, 0.9467, 0.9695, 0.9086, 0.9849],
    'KS': [0.8869, 0.8656, 0.8932, 0.8423, 0.8055, 0.8138, 0.6760, 0.7649, 0.8329, 0.6543, 0.9271],
    'G-mean': [0.9269, 0.9119, 0.9345, 0.9114, 0.8887, 0.9003, 0.8379, 0.8580, 0.9162, 0.8095, 0.9628]
}
# Load the data
data1 = pd.DataFrame(data1)
data2 = pd.DataFrame(data2)

# Extract the data to be compared
metrics = ['Acc', 'AUC', 'KS', 'G-mean']
data1_metrics = data1[metrics]
data2_metrics = data2[metrics]

# Calculate the paired differences
paired_differences = data1_metrics - data2_metrics

# Calculate the mean of the paired differences
mean_differences = paired_differences.mean()

# Calculate the standard deviation of the paired differences
std_differences = paired_differences.std()

# Calculate the t-statistic
t_statistic = mean_differences / (std_differences / np.sqrt(len(paired_differences)))

# Calculate the p-value
p_value = ttest_rel(paired_differences).pvalue

# Interpret the results
for metric, t_statistic, p_value in zip(metrics, t_statistic, p_value):
    print(f"For {metric}, the t-statistic is {t_statistic} and the p-value is {p_value}.")