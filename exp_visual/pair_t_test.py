"""
有两个数据集上11个模型的实验结果如下：

数据集1结果：
Model	Acc	AUC	KS	G-mean
RF	0.8310 	0.9082 	0.6928 	0.8240
ET	0.7895 	0.8828 	0.6257 	0.7606
XGBoost	0.8560 	0.9124 	0.7135 	0.8491
LogR	0.8172 	0.8730 	0.6545 	0.8033
SVM	0.7313 	0.7827 	0.4508 	0.7132
MLP	0.7202 	0.7794 	0.4241 	0.6935
KNN	0.6233 	0.6331 	0.2088 	0.5676
GBDT	0.8449 	0.9078 	0.6970 	0.8355
Bagging	0.7729 	0.8740 	0.5645 	0.7746
AdaBoost	0.8061 	0.8824 	0.6425 	0.7931
MEDBN (ours)	0.8781 	0.9391 	0.7607 	0.9020


数据集2结果：
	Acc	AUC	KS	G-mean
RF	0.9386 	0.9808 	0.8869 	0.9269
ET	0.9289 	0.9768 	0.8656 	0.9119
XGBoost	0.9431 	0.9815 	0.8932 	0.9345
LogR	0.9237 	0.9717 	0.8423 	0.9114
SVM	0.9034 	0.9546 	0.8055 	0.8887
MLP	0.9132 	0.9552 	0.8138 	0.9003
KNN	0.8413 	0.9039 	0.6760 	0.8379
GBDT	0.8840 	0.9467 	0.7649 	0.8580
Bagging	0.9229 	0.9695 	0.8329 	0.9162
AdaBoost	0.8331 	0.9086 	0.6543 	0.8095
MEDBN (ours)	0.9693 	0.9849 	0.9271 	0.9762


对上述结果的各个指标进行配对t检验，要求检验结果能够保存为下表形式的Excel，给出Python代码

Table 3.8 Paired t-test results for different classifiers in terms of G-mean
	VS-FC-PCA	VS-FC	300VS	100VS	50VS	SS
VS-FC-RFFS	0.0023***	0.0484**	0.0066***	0.1122	0.4949	0.0061***
VS-FC-PCA		0.0015***	0.0102**	0.0041***	0.0095***	0.4017
VS-FC			0.0319**	0.6095	0.5758	0.009***
300VS				0.0695*	0.0332**	0.1136
100VS					0.1128	0.0155**
50VS						0.0052***
注：*表示 10%显著水平，**表示 5%显著水平，***表示 1%显著水平。

"""
import pandas as pd
from scipy.stats import ttest_rel

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
# 进行配对t检验
t_test_results = pd.DataFrame(index=df1.index, columns=df1.index)
for i in df1.index:
    for j in df1.index:
        if i == j:
            t_test_results.at[i, j] = '-'
        else:
            a = df1.loc[i]
            b = df2.loc[j]
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

# 输出结果
print(t_test_results)
