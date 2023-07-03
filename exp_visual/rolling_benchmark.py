import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
# 数据
data1 = """
2013	RF	0.7786 	0.8630 	0.5756 	0.7417 	0.9119 	0.6033 
2013	ET	0.7429 	0.8670 	0.5910 	0.6891 	0.9119 	0.5207 
2013	XGBoost	0.7607 	0.8346 	0.5684 	0.7071 	0.9308 	0.5372 
2013	LogR	0.6679 	0.8241 	0.5325 	0.5200 	0.9623 	0.2810 
2013	SVM	0.7071 	0.7981 	0.4990 	0.6529 	0.8742 	0.4876 
2013	MLP	0.6786 	0.7519 	0.4074 	0.6070 	0.8742 	0.4215 
2013	KNN	0.6107 	0.6437 	0.2425 	0.5026 	0.8491 	0.2975 
2013	GBDT	0.7786 	0.8670 	0.5826 	0.7534 	0.8805 	0.6446 
2013	Bagging	0.7250 	0.7928 	0.4676 	0.7064 	0.8050 	0.6198 
2013	AdaBoost	0.7393 	0.8143 	0.4913 	0.7283 	0.7925 	0.6694 
2013	MEDBN	0.7964 	0.8604 	0.6351 	0.8321 	0.7107 	0.8616 
2014	RF	0.8450 	0.9150 	0.7199 	0.8327 	0.8878 	0.7810 
2014	ET	0.8041 	0.8614 	0.6250 	0.7631 	0.9171 	0.6350 
2014	XGBoost	0.8363 	0.9030 	0.6833 	0.8273 	0.8683 	0.7883 
2014	LogR	0.7573 	0.8482 	0.5473 	0.7396 	0.8146 	0.6715 
2014	SVM	0.6930 	0.7525 	0.3740 	0.6478 	0.8098 	0.5182 
2014	MLP	0.7076 	0.7423 	0.3984 	0.6912 	0.7610 	0.6277 
2014	KNN	0.6140 	0.6231 	0.1764 	0.5373 	0.7756 	0.3723 
2014	GBDT	0.8538 	0.9106 	0.7077 	0.8428 	0.8927 	0.7956 
2014	Bagging	0.7690 	0.8235 	0.5493 	0.7426 	0.8488 	0.6496 
2014	AdaBoost	0.8070 	0.8472 	0.5885 	0.7805 	0.8878 	0.6861 
2014	MEDBN	0.8567 	0.8897 	0.6957 	0.8906 	0.7737 	0.9122 
2015	RF	0.8037 	0.8807 	0.6103 	0.7725 	0.8764 	0.6810 
2015	ET	0.7991 	0.8739 	0.5932 	0.7277 	0.9382 	0.5644 
2015	XGBoost	0.8105 	0.8882 	0.6316 	0.7932 	0.8545 	0.7362 
2015	LogR	0.7900 	0.8502 	0.5476 	0.7610 	0.8582 	0.6748 
2015	SVM	0.6849 	0.7488 	0.3948 	0.6610 	0.7418 	0.5890 
2015	MLP	0.6872 	0.7464 	0.3719 	0.6607 	0.7491 	0.5828 
2015	KNN	0.6164 	0.6135 	0.1560 	0.5394 	0.7527 	0.3865 
2015	GBDT	0.7991 	0.8782 	0.5928 	0.7831 	0.8400 	0.7301 
2015	Bagging	0.7580 	0.8292 	0.4996 	0.7491 	0.7818 	0.7178 
2015	AdaBoost	0.7854 	0.8503 	0.5453 	0.7631 	0.8400 	0.6933 
2015	MEDBN	0.8425 	0.9007 	0.7012 	0.8796 	0.7853 	0.8764 
2016	RF	0.7560 	0.8603 	0.5857 	0.7586 	0.7904 	0.7282 
2016	ET	0.7694 	0.8549 	0.5522 	0.7712 	0.7904 	0.7524 
2016	XGBoost	0.7748 	0.8648 	0.5820 	0.7751 	0.7784 	0.7718 
2016	LogR	0.7131 	0.8062 	0.5082 	0.7043 	0.6467 	0.7670 
2016	SVM	0.6810 	0.7235 	0.3800 	0.6778 	0.6527 	0.7039 
2016	MLP	0.6729 	0.7269 	0.3498 	0.6602 	0.5868 	0.7427 
2016	KNN	0.5657 	0.6638 	0.2448 	0.5229 	0.8802 	0.3107 
2016	GBDT	0.7828 	0.8612 	0.5877 	0.7812 	0.7665 	0.7961 
2016	Bagging	0.7292 	0.7921 	0.4640 	0.7197 	0.6587 	0.7864 
2016	AdaBoost	0.7614 	0.8404 	0.5427 	0.7601 	0.7485 	0.7718 
2016	MEDBN	0.8391 	0.8991 	0.6904 	0.8288 	0.8738 	0.7964 
"""

data2 = """
RF	0.8753 	0.9405 	0.7403 	0.8646 	0.9036 	0.8274 
ET	0.8371 	0.9107 	0.6613 	0.7963 	0.9273 	0.6839 
XGBoost	0.8695 	0.9460 	0.7379 	0.8597 	0.8956 	0.8251 
LogR	0.8005 	0.8643 	0.5721 	0.7510 	0.9049 	0.6233 
SVM	0.7980 	0.8580 	0.5812 	0.7818 	0.8388 	0.7287 
MLP	0.7855 	0.8320 	0.5337 	0.7604 	0.8454 	0.6839 
KNN	0.7465 	0.7979 	0.4458 	0.7041 	0.8375 	0.5919 
GBDT	0.7623 	0.9334 	0.7366 	0.7860 	0.6592 	0.9372 
Bagging	0.7648 	0.8800 	0.6611 	0.7822 	0.7015 	0.8722 
AdaBoost	0.8146 	0.8888 	0.6225 	0.8028 	0.8454 	0.7623 
MEDBN	0.8753 	0.9294 	0.7512 	0.9018 	0.8677 	0.8798 
RF	0.8820 	0.9478 	0.7561 	0.8570 	0.9207 	0.7978 
ET	0.8744 	0.9360 	0.7195 	0.8337 	0.9344 	0.7440 
XGBoost	0.8826 	0.9356 	0.7455 	0.8656 	0.9096 	0.8237 
LogR	0.8703 	0.9058 	0.7260 	0.8485 	0.9045 	0.7959 
SVM	0.7967 	0.8568 	0.5741 	0.7727 	0.8338 	0.7161 
MLP	0.7996 	0.8471 	0.5557 	0.7710 	0.8431 	0.7050 
KNN	0.7512 	0.7874 	0.4252 	0.7050 	0.8167 	0.6085 
GBDT	0.8727 	0.9327 	0.7564 	0.8440 	0.9165 	0.7774 
Bagging	0.8178 	0.8911 	0.6667 	0.8169 	0.8193 	0.8145 
AdaBoost	0.8394 	0.8870 	0.6546 	0.8190 	0.8713 	0.7699 
MEDBN	0.9293 	0.9481 	0.8534 	0.9487 	0.9109 	0.9378 
RF	0.8341 	0.9076 	0.6719 	0.8046 	0.8825 	0.7337 
ET	0.8514 	0.9119 	0.6631 	0.7945 	0.9371 	0.6735 
XGBoost	0.8508 	0.9146 	0.6854 	0.8210 	0.8998 	0.7491 
LogR	0.8492 	0.9128 	0.7087 	0.8238 	0.8916 	0.7612 
SVM	0.8084 	0.8454 	0.5570 	0.7588 	0.8841 	0.6512 
MLP	0.8034 	0.8470 	0.5569 	0.7367 	0.8998 	0.6031 
KNN	0.7458 	0.7656 	0.4196 	0.6590 	0.8626 	0.5034 
GBDT	0.8263 	0.8912 	0.6362 	0.7913 	0.8825 	0.7096 
Bagging	0.8061 	0.8668 	0.6041 	0.8020 	0.8137 	0.7904 
AdaBoost	0.8078 	0.8617 	0.5832 	0.7738 	0.8626 	0.6942 
MEDBN	0.9050 	0.9325 	0.8004 	0.9336 	0.8127 	0.9495 
RF	0.8268 	0.9166 	0.6656 	0.8269 	0.8248 	0.8291 
ET	0.8395 	0.9232 	0.6992 	0.8352 	0.8738 	0.7983 
XGBoost	0.8153 	0.9123 	0.6614 	0.8184 	0.7664 	0.8739 
LogR	0.8382 	0.9165 	0.6805 	0.8379 	0.8411 	0.8347 
SVM	0.7962 	0.8797 	0.6151 	0.7943 	0.8131 	0.7759 
MLP	0.7860 	0.8731 	0.5834 	0.7827 	0.8131 	0.7535 
KNN	0.7172 	0.8100 	0.4445 	0.7193 	0.6893 	0.7507 
GBDT	0.7936 	0.8886 	0.6119 	0.7967 	0.7453 	0.8515 
Bagging	0.7019 	0.7940 	0.4624 	0.7024 	0.5911 	0.8347 
AdaBoost	0.7745 	0.8620 	0.5871 	0.7777 	0.7103 	0.8515 
MEDBN	0.8854 	0.9339 	0.7697 	0.8964 	0.8571 	0.9089 

"""

# 解析数据
data_lines = data1.strip().split("\n")
parsed_data = [line.split() for line in data_lines]
years = [int(line[0]) for line in parsed_data]
models = [line[1] for line in parsed_data]
acc = [float(line[2]) for line in parsed_data]
auc = [float(line[3]) for line in parsed_data]
ks = [float(line[4]) for line in parsed_data]
g_mean = [float(line[5]) for line in parsed_data]

unique_models = sorted(list(set(models)))

# 为每个指标分配数字编码
metric_indices = {'ACC': 0, 'AUC': 1, 'KS': 2, 'G-mean': 3}

# 创建一个字典，存储每个模型的数据
data_dict = {}
for model in unique_models:
    data_dict[model] = {'ACC': [], 'AUC': [], 'KS': [], 'G-mean': []}

# 将数据添加到字典中
for i in range(len(models)):
    model = models[i]
    year = years[i]
    data_dict[model]['ACC'].append((year, acc[i]))
    data_dict[model]['AUC'].append((year, auc[i]))
    data_dict[model]['KS'].append((year, ks[i]))
    data_dict[model]['G-mean'].append((year, g_mean[i]))

# 为不同的模型分配不同的线型和标记
model_styles = {
    model: (line_style, marker, color)
    for model, line_style, marker, color in zip(
        unique_models,
        ['-', '--', '-.', ':', '-', '--', '-', ':', '-', '--', '-.'],
        ['o', 's', 'D', 'v', '^', '<', '*', 'x', 'p', 'h', '+'],
        [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            'red', '#7f7f7f', '#bcbd22', '#17becf', '#40004b'
        ],
    )
}

# 设置绘图风格
sns.set(style="whitegrid", )
# custom_palette = sns.color_palette(["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
# sns.set_palette(custom_palette)

# 创建一个画布和子图
fig, ax = plt.subplots(2, 2, figsize=(8, 8), )
metric_names = ['ACC', 'AUC', 'KS', 'G-mean']

# 为每个指标绘制折线图
for model, data in data_dict.items():
    line_style, marker, color = model_styles[model]
    for metric_name, idx in metric_indices.items():
        i, j = divmod(idx, 2)
        x = [item[0] for item in data[metric_name]]
        y = [item[1] for item in data[metric_name]]
        ax[i, j].plot(
            x, y, label=model, linestyle=line_style, marker=marker, markersize=7, color=color
        )
        ax[i, j].set_xlabel("Test year")
        ax[i, j].set_ylabel(metric_name)
        ax[i, j].set_xticks(years)
        ax[i, j].xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax[i, j].xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax[i, j].set_title(f"({chr(ord('a') + idx)})", loc="center", y=-0.25, fontweight='bold')

# 添加全局图例
handles, labels = ax[0, 0].get_legend_handles_labels()
# 自定义标签顺序
custom_labels_order = ['AdaBoost', 'Bagging', 'ET', 'GBDT', 'KNN', 'LogR', 'MLP', 'RF', 'SVM', 'XGBoost', 'MEDBN']
# 根据自定义顺序重新组织 handles 和 labels
ordered_handles = []
ordered_labels = []

for label in custom_labels_order:
    index = labels.index(label)
    ordered_handles.append(handles[index])
    if labels[index] == 'MEDBN':
        ordered_labels.append('MEDBN (ours)')
    else:
        ordered_labels.append(labels[index])

# 使用重排序的 handles 和 labels 创建图例
fig.legend(
    ordered_handles,
    ordered_labels,
    loc="center right",
    bbox_to_anchor=(1., 0.5),
    ncol=1,
    fontsize="small",
)

# plt.xticks(years, weight='normal' )
plt.tight_layout()
plt.show()
