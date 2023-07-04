import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt

# 输入数据集1和数据集2的结果
data1 = {
    'Model': ['RF', 'ET', 'XGBoost', 'LogR', 'SVM', 'MLP', 'KNN', 'GBDT', 'Bagging', 'AdaBoost', 'MEDBN (ours)'],
    'Acc': [0.8310, 0.7895, 0.8560, 0.8172, 0.7313, 0.7202, 0.6233, 0.8449, 0.7729, 0.8061, 0.8781],
    'AUC': [0.9082, 0.8828, 0.9124, 0.8730, 0.7827, 0.7794, 0.6331, 0.9078, 0.8740, 0.8824, 0.9391],
    'KS': [0.6928, 0.6257, 0.7135, 0.6545, 0.4508, 0.4241, 0.2088, 0.6970, 0.5645, 0.6425, 0.7607],
    'G-mean': [0.8240, 0.7606, 0.8491, 0.8033, 0.7132, 0.6935, 0.5676, 0.8355, 0.7746, 0.7931, 0.9020]
}

data2 = {
    'Model': ['RF', 'ET', 'XGBoost', 'LogR', 'SVM', 'MLP', 'KNN', 'GBDT', 'Bagging', 'AdaBoost', 'MEDBN (ours)'],
    'Acc': [0.9386, 0.9289, 0.9431, 0.9237, 0.9034, 0.9132, 0.8413, 0.8840, 0.9229, 0.8331, 0.9693],
    'AUC': [0.9808, 0.9768, 0.9815, 0.9717, 0.9546, 0.9552, 0.9039, 0.9467, 0.9695, 0.9086, 0.9849],
    'KS': [0.8869, 0.8656, 0.8932, 0.8423, 0.8055, 0.8138, 0.6760, 0.7649, 0.8329, 0.6543, 0.9271],
    'G-mean': [0.9269, 0.9119, 0.9345, 0.9114, 0.8887, 0.9003, 0.8379, 0.8580, 0.9162, 0.8095, 0.9762]
}

# 创建数据表
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df1.index = df1['Model']
df2.index = df2['Model']
df1.drop(columns='Model', inplace=True)
df2.drop(columns='Model', inplace=True)

# 计算差值
diff_df = df2 - df1

# 进行配对t检验
t_test_results = pd.DataFrame(index=diff_df.index, columns=diff_df.index)
for i in diff_df.index:
    for j in diff_df.index:
        if i == j:
            t_test_results.at[i, j] = '-'
        else:
            a = diff_df.loc[i]
            b = diff_df.loc[j]
            t_stat, p_value = ttest_rel(a, b)
            if p_value < 0.01:
                sig_level = '***'
            elif p_value < 0.05:
                sig_level = '**'
            elif p_value < 0.1:
                sig_level = '*'
            else:
                sig_level = ''
            t_test_results.at[i, j] = f'{p_value:.4f}{sig_level}'

# 将结果保存到Excel文件中
t_test_results.to_excel('paired_t_test_results.xlsx')

# 创建一个新的DataFrame，仅包含数值型P值
numeric_p_values = t_test_results.applymap(lambda x: float(x.rstrip('***')) if x != '-' else None)

# 将斜对角线的值设置为1
for idx in numeric_p_values.index:
    numeric_p_values.at[idx, idx] = 1.0

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_p_values, annot=True, cmap="YlGnBu_r", vmin=0, vmax=1, cbar_kws={'label': 'P-value'})
plt.title("Paired T-test Results")
plt.tight_layout()
plt.savefig('paired_t_test_heatmap.png')
plt.show()