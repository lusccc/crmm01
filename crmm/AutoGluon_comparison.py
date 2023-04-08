""" Example script for predicting columns of tables, demonstrating simple use-case """

from autogluon.tabular import TabularDataset, TabularPredictor

from autogluon.tabular import TabularDataset
from autogluon.tabular.learner.default_learner import DefaultLearner

# train_data = TabularDataset('../data/cr_sec_6/train.csv')
# val_data = TabularDataset('../data/cr_sec_6/test.csv')
#
# tp = TabularPredictor(label='Rating', path='./aa')
#
#
# learner = tp._learner
# processed_train_data, y_train, processed_val_data, y_val, processed_unlabeled_data, _, _, _ = \
#     learner.general_data_processing(train_data, val_data, None, 0.2, 5)
#
# print()
num_cols = ['currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding', 'netProfitMargin',
            'pretaxProfitMargin', 'grossProfitMargin', 'operatingProfitMargin', 'returnOnAssets',
            'returnOnCapitalEmployed', 'returnOnEquity', 'assetTurnover', 'fixedAssetTurnover', 'debtEquityRatio',
            'debtRatio', 'effectiveTaxRate', 'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare',
            'cashPerShare', 'companyEquityMultiplier', 'ebitPerRevenue', 'enterpriseValueMultiple',
            'operatingCashFlowPerShare', 'operatingCashFlowSalesRatio', 'payablesTurnover']
cat_cols = ['Name', 'Symbol', 'Rating Agency Name', 'Sector', ]
text_cols = ['secKeywords']
use_cols = ['Rating'] + num_cols + cat_cols + text_cols
# used_cols = slice(None, None, None)
# Training time:
train_data = TabularDataset('../data/cr_sec_6/train.csv')[use_cols]
val_data = TabularDataset('../data/cr_sec_6/val.csv')[use_cols]
# train_data = train_data.head(500)  # subsample for faster demo
# print(train_data.head())
label = 'Rating'  # specifies which column do we want to predict
save_path = 'ag_models/'  # where to save trained models
predictor = TabularPredictor(label=label, path=save_path, ).fit(train_data, tuning_data=val_data, num_cpus=24)
# predictor = TabularPredictor(label=label, path=save_path, ).fit(train_data, tuning_data=None, num_cpus=24)
print()

# NOTE: Default settings above are intended to ensure reasonable runtime at the cost of accuracy. To maximize predictive accuracy, do this instead:
# predictor = TabularPredictor(label=label, eval_metric=YOUR_METRIC_NAME, path=save_path).fit(train_data, presets='best_quality')
results = predictor.fit_summary(show_plot=True)

# Inference time:
test_data = TabularDataset('../data/cr_sec_6/test.csv')[use_cols]  # another Pandas DataFrame
# test_data = TabularDataset('../data/cr_sec_6/val.csv')  # another Pandas DataFrame
y_test = test_data[label]
test_data = test_data.drop(labels=[label],
                           axis=1)  # delete labels from test data since we wouldn't have them in practice
print(test_data.head())

predictor = TabularPredictor.load(
    save_path)  # Unnecessary, we reload predictor just to demonstrate how to load previously-trained predictor from file
y_pred = predictor.predict(test_data)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
