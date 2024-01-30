import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel

"""
FOR RANDOM SPLIT DATASET EXP RESULTS
两个数据集：CCR-S和CCR-L的结果对应data1和data2
"""

# 输入数据集1和数据集2的结果
data1 = {
    'Model': ['RF', 'ET', 'XGBoost', 'LogR', 'SVM', 'MLP', 'KNN', 'GBDT', 'Bagging', 'AdaBoost', 'MEDBN (ours)'],
    'Acc': [0.8310, 0.7895, 0.8560, 0.8172, 0.7313, 0.7202, 0.6233, 0.8449, 0.7729, 0.8061, 0.8781],
    'AUC': [0.9082, 0.8828, 0.9124, 0.8730, 0.7827, 0.7794, 0.6331, 0.9078, 0.8740, 0.8824, 0.9391],
    'KS': [0.6928, 0.6257, 0.7135, 0.6545, 0.4508, 0.4241, 0.2088, 0.6970, 0.5645, 0.6425, 0.7607],
    'G-mean': [0.8240, 0.7606, 0.8491, 0.8033, 0.7132, 0.6935, 0.5676, 0.8355, 0.7746, 0.7931, 0.8673] # 0.9020 correct to 0.8673
}

data2 = {
    'Model': ['RF', 'ET', 'XGBoost', 'LogR', 'SVM', 'MLP', 'KNN', 'GBDT', 'Bagging', 'AdaBoost', 'MEDBN (ours)'],
    'Acc': [0.9386, 0.9289, 0.9431, 0.9237, 0.9034, 0.9132, 0.8413, 0.8840, 0.9229, 0.8331, 0.9693],
    'AUC': [0.9808, 0.9768, 0.9815, 0.9717, 0.9546, 0.9552, 0.9039, 0.9467, 0.9695, 0.9086, 0.9849],
    'KS': [0.8869, 0.8656, 0.8932, 0.8423, 0.8055, 0.8138, 0.6760, 0.7649, 0.8329, 0.6543, 0.9271],
    'G-mean': [0.9269, 0.9119, 0.9345, 0.9114, 0.8887, 0.9003, 0.8379, 0.8580, 0.9162, 0.8095, 0.9628] # 0.9762 correct to 0.9628
}

# 创建数据表
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df1.index = df1['Model']
df2.index = df2['Model']
df1.drop(columns='Model', inplace=True)
df2.drop(columns='Model', inplace=True)

"""
是的，计算 `diff_df = df2 - df1` 是合理的。这里，`df1` 和 `df2` 是两个包含相同模型性能指标的数据框，
它们分别表示模型在数据集1和数据集2上的表现。通过对这两个数据框进行逐元素相减，我们可以得到一个新的数据框 `diff_df`，
它表示每个模型在两个数据集上的性能指标的差异。

在这个例子中，`diff_df` 的每一行都表示一个模型在两个数据集上的 Acc、AUC、KS 和 G-mean 等性能指标的差值。
计算这些差值是为了接下来进行配对 t 检验，以检验两个数据集上的模型性能差异是否具有统计显著性。
"""
# 计算差值
diff_df = (df2 + df1)/2

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

# 设置图形的尺寸和样式
plt.figure(figsize=(10, 8))
sns.set(style="white")

# 使用seaborn.heatmap()绘制热力图
ax = sns.heatmap(
    numeric_p_values,
    annot=True,  # 在每个单元格中添加注释
    cmap="GnBu",  # 选择颜色映射
    vmin=0,  # 设置颜色条的最小值
    vmax=0.1,  # 设置颜色条的最大值
    linewidths=0.5,  # 设置单元格之间的间隔线宽度
    cbar_kws={"shrink": 0.5},  # 设置颜色条尺寸
    # mask=np.tril(np.ones_like(numeric_p_values, dtype=bool)) # 仅显示右上三角矩阵

)

# 设置图形的标题、坐标轴标签等
# ax.set_title("Paired T-Test P-Values Heatmap ", )
ax.set_xlabel("", )
ax.set_ylabel("", )

ax.tick_params(axis='both', which='major', labelsize=14)
# 添加color bar描述
cbar = ax.collections[0].colorbar
cbar.set_label('Paired t-test p-value', rotation=270, labelpad=20)
cbar.ax.yaxis.label.set_size(14)

# 显示或保存图形
plt.show()
# plt.savefig("heatmap_filtered.png")
