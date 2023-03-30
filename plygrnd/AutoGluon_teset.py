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
used_cols = ['Rating'] + num_cols
# Training time:
train_data = TabularDataset('../data/cr_sec_6/train.csv')[used_cols]  # can be local CSV file as well, returns Pandas DataFrame
val_data = TabularDataset('../data/cr_sec_6/val.csv')[used_cols]  # can be local CSV file as well, returns Pandas DataFrame
# train_data = train_data.head(500)  # subsample for faster demo
print(train_data.head())
label = 'Rating'  # specifies which column do we want to predict
save_path = 'ag_models/'  # where to save trained models
# TODO NO val set, how to determine stop training
predictor = TabularPredictor(label=label, path=save_path, ).fit(train_data, tuning_data=val_data, num_cpus=24)
print()

# NOTE: Default settings above are intended to ensure reasonable runtime at the cost of accuracy. To maximize predictive accuracy, do this instead:
# predictor = TabularPredictor(label=label, eval_metric=YOUR_METRIC_NAME, path=save_path).fit(train_data, presets='best_quality')
results = predictor.fit_summary(show_plot=True)

# Inference time:
test_data = TabularDataset('../data/cr_sec_6/test.csv')  # another Pandas DataFrame
y_test = test_data[label]
test_data = test_data.drop(labels=[label],
                           axis=1)  # delete labels from test data since we wouldn't have them in practice
print(test_data.head())

predictor = TabularPredictor.load(
    save_path)  # Unnecessary, we reload predictor just to demonstrate how to load previously-trained predictor from file
y_pred = predictor.predict(test_data)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
"""
no tuning data results:
*** Summary of fit() ***
Estimated performance of each model:
                  model  score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
0              CatBoost   0.682540       0.326438  329.986671                0.326438         329.986671            1       True          8
1   WeightedEnsemble_L2   0.682540       0.326728  330.173421                0.000290           0.186750            2       True         14
2       NeuralNetFastAI   0.642857       0.008096    2.879750                0.008096           2.879750            1       True          3
3            LightGBMXT   0.642857       0.038630   32.928829                0.038630          32.928829            1       True          4
4              LightGBM   0.615079       0.036061   35.032663                0.036061          35.032663            1       True          5
5               XGBoost   0.591270       0.046652   23.128212                0.046652          23.128212            1       True         11
6         LightGBMLarge   0.583333       0.091743   82.265570                0.091743          82.265570            1       True         13
7      RandomForestEntr   0.579365       0.068071    3.397269                0.068071           3.397269            1       True          7
8      RandomForestGini   0.567460       0.071167    3.649547                0.071167           3.649547            1       True          6
9        NeuralNetTorch   0.547619       0.026814    4.141184                0.026814           4.141184            1       True         12
10       ExtraTreesGini   0.547619       0.072956    3.905140                0.072956           3.905140            1       True          9
11       ExtraTreesEntr   0.531746       0.071801    3.492379                0.071801           3.492379            1       True         10
12       KNeighborsDist   0.234127       0.257423    1.787576                0.257423           1.787576            1       True          2
13       KNeighborsUnif   0.234127       0.376220    1.727853                0.376220           1.727853            1       True          1
"""
"""
has tuning data results:
Estimated performance of each model:
                  model  score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
0   WeightedEnsemble_L2   0.733333       0.796528  285.584523                0.000344           0.176498            2       True         14
1       NeuralNetFastAI   0.716667       0.008135    3.223203                0.008135           3.223203            1       True          3
2              CatBoost   0.694444       0.315869  207.370804                0.315869         207.370804            1       True          8
3              LightGBM   0.644444       0.045327   43.664675                0.045327          43.664675            1       True          5
4        NeuralNetTorch   0.627778       0.018964    7.213622                0.018964           7.213622            1       True         12
5            LightGBMXT   0.627778       0.036973   38.657798                0.036973          38.657798            1       True          4
6      RandomForestGini   0.627778       0.075965    3.784702                0.075965           3.784702            1       True          6
7         LightGBMLarge   0.616667       0.034858  251.796986                0.034858         251.796986            1       True         13
8      RandomForestEntr   0.611111       0.068888    3.505581                0.068888           3.505581            1       True          7
9        ExtraTreesGini   0.600000       0.072077    3.961279                0.072077           3.961279            1       True          9
10              XGBoost   0.594444       0.032525   30.367700                0.032525          30.367700            1       True         11
11       ExtraTreesEntr   0.577778       0.068814    3.591933                0.068814           3.591933            1       True         10
12       KNeighborsDist   0.266667       0.330606    1.827240                0.330606           1.827240            1       True          2
13       KNeighborsUnif   0.227778       0.327494    1.807583                0.327494           1.807583            1       True          1
"""

"""
only num cols and has tuning data:
*** Summary of fit() ***
Estimated performance of each model:
                  model  score_val  pred_time_val  fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
0   WeightedEnsemble_L2   0.611111       0.137387  4.458234                0.000283           0.268009            2       True         14
1        ExtraTreesGini   0.583333       0.061308  0.573086                0.061308           0.573086            1       True          9
2              LightGBM   0.577778       0.005023  2.024295                0.005023           2.024295            1       True          5
3         LightGBMLarge   0.561111       0.002542  3.720434                0.002542           3.720434            1       True         13
4            LightGBMXT   0.561111       0.004037  2.518531                0.004037           2.518531            1       True          4
5               XGBoost   0.561111       0.005631  1.757163                0.005631           1.757163            1       True         11
6      RandomForestGini   0.555556       0.060399  0.604215                0.060399           0.604215            1       True          6
7        ExtraTreesEntr   0.550000       0.061080  0.423410                0.061080           0.423410            1       True         10
8              CatBoost   0.538889       0.001964  1.480896                0.001964           1.480896            1       True          8
9      RandomForestEntr   0.538889       0.060823  0.433239                0.060823           0.433239            1       True          7
10       NeuralNetTorch   0.522222       0.013008  1.703002                0.013008           1.703002            1       True         12
11       KNeighborsDist   0.394444       0.025141  0.011376                0.025141           0.011376            1       True          2
12      NeuralNetFastAI   0.388889       0.004104  2.576390                0.004104           2.576390            1       True          3
13       KNeighborsUnif   0.366667       0.072247  0.011648                0.072247           0.011648            1       True          1
"""
