""" Example script for predicting columns of tables, demonstrating simple use-case """

from autogluon.tabular import TabularDataset, TabularPredictor

from autogluon.tabular import TabularDataset

# dataset_name = 'cr'
dataset_name = 'cr2'

num_cols = ['currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding', 'netProfitMargin',
            'pretaxProfitMargin', 'grossProfitMargin', 'operatingProfitMargin', 'returnOnAssets',
            'returnOnCapitalEmployed', 'returnOnEquity', 'assetTurnover', 'fixedAssetTurnover', 'debtEquityRatio',
            'debtRatio', 'effectiveTaxRate', 'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare',
            'cashPerShare', 'companyEquityMultiplier', 'ebitPerRevenue', 'enterpriseValueMultiple',
            'operatingCashFlowPerShare', 'operatingCashFlowSalesRatio', 'payablesTurnover']
cat_cols = ['Name', 'Symbol', 'Rating Agency Name', 'Sector', ]
text_cols = ['secKeywords']
use_cols = ['Rating'] + num_cols + cat_cols + text_cols
train_data = TabularDataset(f'../data/{dataset_name}_sec_6/train.csv')[use_cols]
val_data = TabularDataset(f'../data/{dataset_name}_sec_6/val.csv')[use_cols]
# train_data = train_data.head(500)  # subsample for faster demo
# print(train_data.head())
label = 'Rating'  # specifies which column do we want to predict
save_path = 'ag_models/'  # where to save trained models

only_test = False

if not only_test:
    predictor = TabularPredictor(label=label, path=save_path, ).fit(train_data, tuning_data=val_data, num_cpus=24,
                                                                    save_space=True)
    # NOTE: Default settings above are intended to ensure reasonable runtime at the cost of accuracy. To maximize predictive accuracy, do this instead:
    # predictor = TabularPredictor(label=label, eval_metric=YOUR_METRIC_NAME, path=save_path).fit(train_data, presets='best_quality')
    results = predictor.fit_summary(show_plot=True)

# Inference time:
test_data = TabularDataset('../data/cr_sec_6/test.csv')[use_cols]  # another Pandas DataFrame
# test_data = TabularDataset('../data/cr_sec_6/val.csv')  # another Pandas DataFrame
y_test = test_data[label]
# delete labels from test data since we wouldn't have them in practice
test_data = test_data.drop(labels=[label], axis=1)

predictor = TabularPredictor.load(save_path)
# y_pred = predictor.predict(test_data)
# perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True, detailed_report=True)

# 对测试集进行预测并评估各个模型
model_performance = {}

for model_name in predictor.get_model_names():
    y_pred_proba = predictor.predict_proba(test_data, model=model_name)
    y_pred = y_pred_proba.idxmax(axis=1)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    model_performance[model_name] = perf

# 输出各模型在测试集上的结果
for model_name, perf in model_performance.items():
    print(f"{model_name}: {perf}")
