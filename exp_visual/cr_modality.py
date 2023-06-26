import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
data = [
    {"Numerical": "yes", "Category": "", "Text": "", "Acc": 0.7618, "AUC": 0.8619, "KS": 0.5994, "G-mean": 0.8208},
    {"Numerical": "", "Category": "yes", "Text": "", "Acc": 0.7922, "AUC": 0.8753, "KS": 0.6080, "G-mean": 0.8431},
    {"Numerical": "", "Category": "", "Text": "yes", "Acc": 0.6454, "AUC": 0.6903, "KS": 0.3176, "G-mean": 0.7549},
    {"Numerical": "yes", "Category": "yes", "Text": "", "Acc": 0.8615, "AUC": 0.9323, "KS": 0.7519, "G-mean": 0.8956},
    {"Numerical": "yes", "Category": "", "Text": "yes", "Acc": 0.7729, "AUC": 0.8776, "KS": 0.6092, "G-mean": 0.8248},
    {"Numerical": "", "Category": "yes", "Text": "yes", "Acc": 0.7839, "AUC": 0.8711, "KS": 0.6106, "G-mean": 0.8431},
    {"Numerical": "yes", "Category": "yes", "Text": "yes", "Acc": 0.8781, "AUC": 0.9391, "KS": 0.7607, "G-mean": 0.9020},
]

# 创建数据框
df = pd.DataFrame(data)

# 设置标签
labels = ['N', 'C', 'T', 'NC', 'NT', 'CT', 'NCT']
metrics = ['Acc', 'AUC', 'KS', 'G-mean']

# 设置绘图风格
plt.style.use('ggplot')

# 创建堆叠条形图
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
opacity = 0.8

index = np.arange(len(labels))
rects1 = plt.bar(index, df['Acc'], bar_width, alpha=opacity, color='b', label='Acc')
rects2 = plt.bar(index + bar_width, df['AUC'], bar_width, alpha=opacity, color='g', label='AUC')
rects3 = plt.bar(index + 2 * bar_width, df['KS'], bar_width, alpha=opacity, color='r', label='KS')
rects4 = plt.bar(index + 3 * bar_width, df['G-mean'], bar_width, alpha=opacity, color='y', label='G-mean')

plt.xlabel('组合')
plt.ylabel('评分')
plt.title('不同组合的实验结果')
plt.xticks(index + 1.5 * bar_width, labels)
plt.legend()

plt.tight_layout()
plt.show()