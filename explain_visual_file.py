import pickle
import numpy as np
from matplotlib import pyplot as plt

def load_obj(fname):
    with open(fname, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

dt = 'cr'

word_importance_pos = load_obj(f'{dt}_word_importance_pos.pkl')
word_importance_neg = load_obj(f'{dt}_word_importance_neg.pkl')
word_freq_pos = load_obj(f'{dt}_word_freq_pos.pkl')
word_freq_neg = load_obj(f'{dt}_word_freq_neg.pkl')

print()

# 平均影响力
for word in word_importance_pos:
    word_importance_pos[word] /= word_freq_pos[word]

for word in word_importance_neg:
    word_importance_neg[word] /= word_freq_neg[word]

# 计算加权重要性，并引入词频门槛
frequency_threshold = 3  # 可以根据具体情况调整这个阈值

weighted_importance_pos = {word: word_importance_pos[word] * word_freq_pos[word]
                           for word in word_importance_pos if word_freq_pos[word] >= frequency_threshold}
weighted_importance_neg = {word: word_importance_neg[word] * word_freq_neg[word]
                           for word in word_importance_neg if word_freq_neg[word] >= frequency_threshold}

# 取加权影响最大的前30个词
top_words_pos = sorted(weighted_importance_pos.items(), key=lambda x: abs(x[1]), reverse=True)[:30]
top_words_neg = sorted(weighted_importance_neg.items(), key=lambda x: abs(x[1]), reverse=True)[:30]

# 可视化
def plot_top_words(top_words, title=None):
    words, importances = zip(*top_words)
    y_pos = np.arange(len(words))

    plt.figure(figsize=(10, 8))
    colors = np.where(np.array(importances) > 0, 'royalblue', 'orange')
    plt.barh(y_pos, importances, align='center', color=colors)
    plt.yticks(y_pos, words)
    plt.xlabel('Weighted Importance')
    plt.ylabel('Word')
    if title:
        plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

plot_top_words(top_words_pos, 'Top 20 words for Positive Sentiment')
plot_top_words(top_words_neg, 'Top 20 words for Negative Sentiment')