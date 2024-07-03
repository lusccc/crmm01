import pickle
from collections import defaultdict

import numpy as np
import shap
import torch
from lime.lime_text import LimeTextExplainer
from matplotlib import pyplot as plt
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from tqdm import tqdm
from transformers import pipeline

from dataset.multimodal_dataset import MultimodalDataset


# https://github.com/shap/shap/blob/master/notebooks/text_examples/sentiment_analysis/Emotion%20classification%20multiclass%20example.ipynb


def shap_visual(model, tokenizer, test_dataset: MultimodalDataset):
    # 统计所有文本中的词频
    texts = []
    for data in test_dataset:
        labels, _, _, text = data.values()
        texts.append(text)

    pred = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        return_all_scores=True,
    )

    explainer = shap.Explainer(pred,
                               # masker=shap.maskers.Text(tokenizer=r"\W+")
                               )
    shap_values = explainer(texts[:10])

    # 使用 text_plot 显示 SHAP 解释
    # shap.plots.text(shap_values)

    shap.plots.bar(shap_values[:, :, "LABEL_0"].mean(0),  # label_0 bad sample
                   # order=shap.Explanation.argsort,
                   max_display=20)
    print()


def lime_visual(model, tokenizer, test_dataset):
    texts = []
    labels = []
    for data in test_dataset:
        label, _, _, text = data.values()
        texts.append(text)
        labels.append(label)

    # LIME解释器
    explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

    # 模型预测函数包装
    def predict_proba(input_text):
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}  # 将输入移到GPU
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()  # 将结果移回CPU
        return probs

    # 自定义停用词表
    custom_stop_words = set(ENGLISH_STOP_WORDS).union(
        {'cms', 'fx', 'nrcs'}
    )

    # 收集LIME解释结果
    word_importance_pos = defaultdict(float)
    word_importance_neg = defaultdict(float)
    word_freq_pos = defaultdict(int)
    word_freq_neg = defaultdict(int)

    for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Processing"):
        exp = explainer.explain_instance(text, predict_proba, num_features=20)
        for word, importance in exp.as_list():
            if word in custom_stop_words:
                continue
            if label == 1:  # 正面
                word_importance_pos[word] += importance
                word_freq_pos[word] += 1
            else:  # 负面
                word_importance_neg[word] += importance
                word_freq_neg[word] += 1

    # 平均影响力
    for word in word_importance_pos:
        word_importance_pos[word] /= word_freq_pos[word]

    for word in word_importance_neg:
        word_importance_neg[word] /= word_freq_neg[word]


    # 取影响最大的前20个词
    top_words_pos = sorted(word_importance_pos.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
    top_words_neg = sorted(word_importance_neg.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

    # 可视化
    def plot_top_words(top_words, title=None):
        words, importances = zip(*top_words)
        y_pos = np.arange(len(words))

        plt.figure(figsize=(10, 8))
        colors = np.where(np.array(importances) > 0, 'royalblue', 'orange')
        plt.barh(y_pos, importances, align='center', color=colors)
        plt.yticks(y_pos, words)
        plt.xlabel('Importance')
        plt.ylabel('Word')  # 设置Y轴标签为"Word"
        if title:
            plt.title(title)
        plt.gca().invert_yaxis()  # 反转Y轴让最高的排在上面
        plt.show()

    plot_top_words(top_words_pos,
                   # 'Top 20 words for Positive Sentiment'
                   )
    plot_top_words(top_words_neg,
                   # 'Top 20 words for Negative Sentiment'
                   )
    print()


"""
手动保存下面这些对象，然后在explain_visual_file_v2.py进行可视化
"""
# word_importance_pos
# word_importance_neg
# word_freq_pos
# word_freq_neg
# with open('data.pkl', 'wb') as file:
#     pickle.dump(data, file)



