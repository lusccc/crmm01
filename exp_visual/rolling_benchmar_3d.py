import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
# 数据
data = """
2013	RF	0.7786 	0.8630 	0.5756 	0.7417  
2013	ET	0.7429 	0.8670 	0.5910 	0.6891
2013	XGBoost	0.7607 	0.8346 	0.5684 	0.7071
2013	LogR	0.6679 	0.8241 	0.5325 	0.5200
2013	SVM	0.7071 	0.7981 	0.4990 	0.6529
2013	MLP	0.6786 	0.7519 	0.4074 	0.6070
2013	KNN	0.6107 	0.6437 	0.2425 	0.5026
2013	MEDBN	0.7964 	0.8604 	0.6351 	0.8321 
2014	RF	0.8450 	0.9150 	0.7199 	0.8327   
2014	ET	0.8041 	0.8614 	0.6250 	0.7631
2014	XGBoost	0.8363 	0.9030 	0.6833 	0.8273
2014	LogR	0.7573 	0.8482 	0.5473 	0.7396
2014	SVM	0.6930 	0.7525 	0.3740 	0.6478
2014	MLP	0.7076 	0.7423 	0.3984 	0.6912
2014	KNN	0.6140 	0.6231 	0.1764 	0.5373   
2014	MEDBN	0.8567 	0.8897 	0.6957 	0.8906    
2015	RF	0.8037 	0.8807 	0.6103 	0.7725
2015	ET	0.7991 	0.8739 	0.5932 	0.7277
2015	XGBoost	0.8105 	0.8882 	0.6316 	0.7932
2015	LogR	0.7900 	0.8502 	0.5476 	0.7610
2015	SVM	0.6849 	0.7488 	0.3948 	0.6610
2015	MLP	0.6872 	0.7464 	0.3719 	0.6607
2015	KNN	0.6164 	0.6135 	0.1560 	0.5394
2015	MEDBN	0.8425 	0.9007 	0.7012 	0.8796  
2016	RF	0.7560 	0.8603 	0.5857 	0.7586  
2016	ET	0.7694 	0.8549 	0.5522 	0.7712
2016	XGBoost	0.7748 	0.8648 	0.5820 	0.7751
2016	LogR	0.7131 	0.8062 	0.5082 	0.7043
2016	SVM	0.6810 	0.7235 	0.3800 	0.6778
2016	MLP	0.6729 	0.7269 	0.3498 	0.6602
2016	KNN	0.5657 	0.6638 	0.2448 	0.5229
2016	MEDBN	0.8391 	0.8991 	0.6904 	0.8288
"""

# 解析数据
data_lines = data.strip().split("\n")
parsed_data = [line.split() for line in data_lines]
years = [int(line[0]) for line in parsed_data]
models = [line[1] for line in parsed_data]
acc = [float(line[2]) for line in parsed_data]
auc = [float(line[3]) for line in parsed_data]
ks = [float(line[4]) for line in parsed_data]
g_mean = [float(line[5]) for line in parsed_data]

unique_models = sorted(list(set(models)))
unique_years = sorted(list(set(years)))

# 创建一个字典，存储每个模型的数据
data_dict_3d = {}
for model in unique_models:
    data_dict_3d[model] = {}
    for year in unique_years:
        data_dict_3d[model][year] = {'ACC': None, 'AUC': None, 'KS': None, 'G-mean': None}

# 将数据添加到字典中
for i in range(len(models)):
    model = models[i]
    year = years[i]
    data_dict_3d[model][year]['ACC'] = acc[i]
    data_dict_3d[model][year]['AUC'] = auc[i]
    data_dict_3d[model][year]['KS'] = ks[i]
    data_dict_3d[model][year]['G-mean'] = g_mean[i]

# 设置绘图风格
sns.set(style="whitegrid", palette="muted")

# 创建一个画布和子图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 定义评估指标的顺序
metrics_order = ['ACC', 'AUC', 'KS', 'G-mean']

# 为每个模型绘制三维散点图并连接数据点
for idx, model in enumerate(unique_models):
    for metric_idx, metric in enumerate(metrics_order):
        x = [year for year in unique_years]
        y = [metric_idx] * len(unique_years)
        z = [data_dict_3d[model][year][metric] for year in unique_years]

        # 绘制散点图
        ax.scatter(x, y, z, label=f"{model}" if metric_idx == 0 else None, s=50, depthshade=True, marker=f"${idx + 1}$")

        # 连接数据点
        ax.plot(x, y, z, linestyle="--", linewidth=1)

# 设置坐标轴标签
ax.set_xlabel("Test year")
ax.set_yticks(range(len(metrics_order)))
ax.set_yticklabels(metrics_order)
ax.set_ylabel("Metrics")
ax.set_zlabel("Score")

# 添加图例
ax.legend()

plt.show()