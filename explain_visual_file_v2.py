import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_obj(fname):
    with open(fname, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


dt = 'cr2'

word_importance_pos = load_obj(f'{dt}_word_importance_pos.pkl')
word_importance_neg = load_obj(f'{dt}_word_importance_neg.pkl')
word_freq_pos = load_obj(f'{dt}_word_freq_pos.pkl')
word_freq_neg = load_obj(f'{dt}_word_freq_neg.pkl')

exclude_words = [
    'venezuela',
    'pharmaceuticals',   # 不需要复数
    'loans',  # 不需要复数
    'data',
    'rate',
    'rates',
    'notes',
    'southwests',
    'epa',
    'fcc'
    # 'income'
]
for word in exclude_words:
    if word in word_importance_pos:
        del word_importance_pos[word]
    if word in word_importance_neg:
        del word_importance_neg[word]

# 平均影响力
for word in word_importance_pos:
    word_importance_pos[word] /= word_freq_pos[word]

for word in word_importance_neg:
    word_importance_neg[word] /= word_freq_neg[word]

# 计算加权重要性，并引入词频门槛
frequency_threshold = 12 if dt == 'cr2' else 6  # 可以根据具体情况调整这个阈值

weighted_importance_pos = {word: word_importance_pos[word] * word_freq_pos[word]
                           for word in word_importance_pos if word_freq_pos[word] >= frequency_threshold}
weighted_importance_neg = {word: word_importance_neg[word] * word_freq_neg[word]
                           for word in word_importance_neg if word_freq_neg[word] >= frequency_threshold}

# 取加权影响最大的前30个词
# top_words_pos = sorted(weighted_importance_pos.items(), key=lambda x: abs(x[1]), reverse=True)[:30]
# top_words_neg = sorted(weighted_importance_neg.items(), key=lambda x: abs(x[1]), reverse=True)[:30]

# 分别取加权影响的正值和负值
n = 10
top_pos_positive = sorted((item for item in weighted_importance_pos.items() if item[1] > 0),
                          key=lambda x: x[1], reverse=True)[:n]
top_pos_negative = sorted((item for item in weighted_importance_pos.items() if item[1] < 0),
                          key=lambda x: x[1])[:n]

top_neg_positive = sorted((item for item in weighted_importance_neg.items() if item[1] > 0),
                          key=lambda x: x[1], reverse=True)[:n]
top_neg_negative = sorted((item for item in weighted_importance_neg.items() if item[1] < 0),
                          key=lambda x: x[1])[:n]

# Combine the results
top_words_pos = top_pos_positive + top_pos_negative
top_words_neg = top_neg_positive + top_neg_negative


def create_colormap():
    return LinearSegmentedColormap.from_list("", ["#FFABAB", "#FFFFFF", "#99FF99"])


def format_importance(value):
    return f"{value:+.3f}"


def plot_top_words(top_words, title, ax):
    words, importances = zip(*top_words)
    print(f'{title}')

    # importances = [format_importance(imp) for imp in importances]
    data = {'Word': words, 'Importance': [format_importance(imp) for imp in importances], 'Importance_ori': importances}
    df = pd.DataFrame(data)
    df = df.sort_values(by='Importance_ori', ascending=False)
    df = df.drop('Importance_ori', axis=1)

    print(df)

    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=["#DEDEDE"] * len(df.columns),
                     colWidths=[0.6, 0.4])

    # 去掉表格边框
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('white')

    # 设置表格颜色
    cmap = create_colormap()
    importances_float = [float(imp) for imp in df['Importance']]
    norm = plt.Normalize(min(importances_float), max(importances_float))

    for i, imp in enumerate(importances_float):
        color = cmap(norm(imp))
        table[(i + 1, 0)].set_facecolor(color)
        table[(i + 1, 1)].set_facecolor(color)

    # 使用 annotate 添加底部标题
    ax.annotate(title, xy=(0.5, 0.13), xycoords='axes fraction', ha='center', va='center', fontsize=12,
                fontweight='bold')


fig, axs = plt.subplots(1, 2, figsize=(9, 10))
plot_top_words(top_words_pos,
               '(a) Important words for predictions of good credit.',
               # '(a) Good credit',
               axs[0])
print()
plot_top_words(top_words_neg,
               '(b) Important words for predictions of bad credit.',
               # '(b) Bad credit',
               axs[1])
plt.tight_layout(pad=3.0)
plt.show()
