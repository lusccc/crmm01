import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product


plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成随机实验结果数据
def generate_random_data():
    data = {}
    for train_year in range(2010, 2014):
        for test_year in range(train_year + 3, 2017):
            key = f"{train_year}-{train_year + 2}/{test_year}"
            data[key] = {
                "acc": np.random.uniform(0.7, 1),
                "ks": np.random.uniform(0.4, 1),
                "gmean": np.random.uniform(0.5, 1),
                "type1_acc": np.random.uniform(0.7, 1),
                "type2_acc": np.random.uniform(0.7, 1),
            }
    return data

# 绘制三维条形图
def plot_3d_bar_chart(dataset, metric):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xpos = np.arange(len(dataset))
    ypos = np.arange(len(dataset))
    zpos = np.zeros(len(dataset))

    dx = np.ones(len(dataset)) * 0.5
    dy = np.ones(len(dataset)) * 0.5
    dz = [d[metric] for d in dataset.values()]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)

    ax.set_xticks(xpos)
    ax.set_xticklabels(dataset.keys(), rotation=45, ha="right")
    ax.set_yticks(ypos)
    ax.set_yticklabels(dataset.keys())

    ax.set_xlabel("Train Years")
    ax.set_ylabel("Test Years")
    ax.set_zlabel(metric)

    plt.title(f"Dataset {dataset_num}: {metric}")
    plt.show()

# 随机生成实验结果数据
dataset1 = generate_random_data()
dataset2 = generate_random_data()

# 为每个数据集绘制三维条形图
for dataset_num, dataset in enumerate([dataset1, dataset2], start=1):
    for metric in ["acc", "ks", "gmean", "type1_acc", "type2_acc"]:
        plot_3d_bar_chart(dataset, metric)