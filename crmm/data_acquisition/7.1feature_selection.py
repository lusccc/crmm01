import pandas as pd
from autogluon.core import TabularDataset
from autogluon.tabular import TabularPredictor

# dataset_name = 'cr'
dataset_name = 'cr2'
data_df = pd.read_csv(f'../../data/{dataset_name}_sec_ori/corporate_rating_with_cik_and_summarized_kw_sec_text.csv',
                      index_col=0)
label_column = 'Rating'

# 将数据转换为AutoGluon的TabularDataset
train_data = TabularDataset(data_df)

# 初始化TabularPredictor
predictor = TabularPredictor(label=label_column, eval_metric='accuracy')

# 使用.fit()方法进行特征选择和模型训练
predictor.fit(train_data, presets='best_quality', time_limit=3600)

# 获取特征重要性
feature_importances = predictor.feature_importance()

# 打印特征重要性
print(feature_importances)
