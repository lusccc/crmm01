import pandas as pd
import matplotlib.pyplot as plt

# 创建数据集
# cr
data = {
    'training_years': ['2010-2012/2013', '2011-2013/2014', '2012-2014/2015', '2013-2015/2016'],
    'training_samples': [367, 640, 938, 1060],
    'testing_year': [2013, 2014, 2015, 2016],
    'testing_samples': [280, 342, 438, 375],
    'train_test_ratio': [1.310714286, 1.871345029, 2.141552511, 2.826666667],
}

# cr2
# data = {
#     'training_years': ['2010-2012/2013', '2011-2013/2014', '2012-2014/2015', '2013-2015/2016'],
#     'training_samples': [1189, 2361, 3798, 4705],
#     'testing_year': [2013, 2014, 2015, 2016],
#     'testing_samples': [1203, 1712, 1790, 785],
#     'train_test_ratio': [0.988362427, 1.379088785, 2.121787709, 5.993630573],
# }

df = pd.DataFrame(data)

# 创建figure和axis对象
fig, ax1 = plt.subplots()

# 绘制训练样本和测试样本的折线图
l1, = ax1.plot(df['training_years'], df['training_samples'], marker='o', label='Training Samples')
l2, = ax1.plot(df['training_years'], df['testing_samples'], marker='*', label='Testing Samples')
ax1.set_xlabel('Train/test Years')
ax1.set_ylabel('Number of Samples')

# 在数据点上显示具体数值
for i, txt in enumerate(df['training_samples']):
    ax1.annotate(txt, (df['training_years'][i], df['training_samples'][i]), textcoords="offset points", xytext=(-5, 3),
                 ha='center')
for i, txt in enumerate(df['testing_samples']):
    ax1.annotate(txt, (df['training_years'][i], df['testing_samples'][i]), textcoords="offset points", xytext=(7, 8),
                 ha='center')

# 创建一个次要的y轴来显示训练/测试比例
ax2 = ax1.twinx()
l3, = ax2.plot(df['training_years'], df['train_test_ratio'], marker='x', color='r', label='Train/Test Ratio')
ax2.set_ylabel('Train/Test Ratio')

# 在数据点上显示具体数值
for i, txt in enumerate(df['train_test_ratio']):
    ax2.annotate(f'{txt:.2f}', (df['training_years'][i], df['train_test_ratio'][i]), textcoords="offset points",
                 xytext=(-7, -10), ha='center')

# 添加图例
fig.legend([l1, l2, l3], ['Training Samples', 'Testing Samples', 'Train/Test Ratio'], loc="upper left",
           bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)

# plt.title('Number of Samples and Train/Test Ratio over Training Years')
plt.show()
