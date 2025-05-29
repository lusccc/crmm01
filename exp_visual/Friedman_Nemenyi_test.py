import pandas as pd
from scipy.stats import friedmanchisquare

from exp_visual.nemenyi_plot_tool import nemenyi_plot_multiple

"""
FOR ROLLING SPLIT DATASET EXP RESULTS
两个数据集rolling的结果，每一个数据集rolling划分可得四个数据集，那么共有8个数据集。
8个数据集的结果我都放在all_results里了
"""

all_results = [
    """
    RF	0.8753 	0.9405 	0.7403 	0.8646 	0.9036 	0.8274 
    ET	0.8371 	0.9107 	0.6613 	0.7963 	0.9273 	0.6839 
    XGBoost	0.8695 	0.9460 	0.7379 	0.8597 	0.8956 	0.8251 
    LogR	0.8005 	0.8643 	0.5721 	0.7510 	0.9049 	0.6233 
    SVM	0.7980 	0.8580 	0.5812 	0.7818 	0.8388 	0.7287 
    MLP	0.7855 	0.8320 	0.5337 	0.7604 	0.8454 	0.6839 
    KNN	0.7465 	0.7979 	0.4458 	0.7041 	0.8375 	0.5919 
    GBDT	0.7623 	0.9334 	0.7366 	0.7860 	0.6592 	0.9372 
    Bagging	0.7648 	0.8800 	0.6611 	0.7822 	0.7015 	0.8722 
    AdaBoost	0.8146 	0.8888 	0.6225 	0.8028 	0.8454 	0.7623 
    MEDBN(ours)	0.8753 	0.9294 	0.7512 	0.8737 	0.8677 	0.8798 
    """,
    """
    RF	0.8820 	0.9478 	0.7561 	0.8570 	0.9207 	0.7978 
    ET	0.8744 	0.9360 	0.7195 	0.8337 	0.9344 	0.7440 
    XGBoost	0.8826 	0.9356 	0.7455 	0.8656 	0.9096 	0.8237 
    LogR	0.8703 	0.9058 	0.7260 	0.8485 	0.9045 	0.7959 
    SVM	0.7967 	0.8568 	0.5741 	0.7727 	0.8338 	0.7161 
    MLP	0.7996 	0.8471 	0.5557 	0.7710 	0.8431 	0.7050 
    KNN	0.7512 	0.7874 	0.4252 	0.7050 	0.8167 	0.6085 
    GBDT	0.8727 	0.9327 	0.7564 	0.8440 	0.9165 	0.7774 
    Bagging	0.8178 	0.8911 	0.6667 	0.8169 	0.8193 	0.8145 
    AdaBoost	0.8394 	0.8870 	0.6546 	0.8190 	0.8713 	0.7699 
    MEDBN(ours)	0.9293 	0.9481 	0.8534 	0.9243 	0.9109 	0.9378 
    """,
    """
    RF	0.8341 	0.9076 	0.6719 	0.8046 	0.8825 	0.7337 
    ET	0.8514 	0.9119 	0.6631 	0.7945 	0.9371 	0.6735 
    XGBoost	0.8508 	0.9146 	0.6854 	0.8210 	0.8998 	0.7491 
    LogR	0.8492 	0.9128 	0.7087 	0.8238 	0.8916 	0.7612 
    SVM	0.8084 	0.8454 	0.5570 	0.7588 	0.8841 	0.6512 
    MLP	0.8034 	0.8470 	0.5569 	0.7367 	0.8998 	0.6031 
    KNN	0.7458 	0.7656 	0.4196 	0.6590 	0.8626 	0.5034 
    GBDT	0.8263 	0.8912 	0.6362 	0.7913 	0.8825 	0.7096 
    Bagging	0.8061 	0.8668 	0.6041 	0.8020 	0.8137 	0.7904 
    AdaBoost	0.8078 	0.8617 	0.5832 	0.7738 	0.8626 	0.6942 
    MEDBN(ours)	0.9050 	0.9325 	0.8004 	0.8785 	0.8127 	0.9495 
    """,
    """
    RF	0.8268 	0.9166 	0.6656 	0.8269 	0.8248 	0.8291 
    ET	0.8395 	0.9232 	0.6992 	0.8352 	0.8738 	0.7983 
    XGBoost	0.8153 	0.9123 	0.6614 	0.8184 	0.7664 	0.8739 
    LogR	0.8382 	0.9165 	0.6805 	0.8379 	0.8411 	0.8347 
    SVM	0.7962 	0.8797 	0.6151 	0.7943 	0.8131 	0.7759 
    MLP	0.7860 	0.8731 	0.5834 	0.7827 	0.8131 	0.7535 
    KNN	0.7172 	0.8100 	0.4445 	0.7193 	0.6893 	0.7507 
    GBDT	0.7936 	0.8886 	0.6119 	0.7967 	0.7453 	0.8515 
    Bagging	0.7019 	0.7940 	0.4624 	0.7024 	0.5911 	0.8347 
    AdaBoost	0.7745 	0.8620 	0.5871 	0.7777 	0.7103 	0.8515 
    MEDBN(ours)	0.8854 	0.9339 	0.7697 	0.8826 	0.8571 	0.9089 
    """,
    """
    RF	0.7786 	0.8630 	0.5756 	0.7417 	0.9119 	0.6033 
    ET	0.7429 	0.8670 	0.5910 	0.6891 	0.9119 	0.5207 
    XGBoost	0.7607 	0.8346 	0.5684 	0.7071 	0.9308 	0.5372 
    LogR	0.6679 	0.8241 	0.5325 	0.5200 	0.9623 	0.2810 
    SVM	0.7071 	0.7981 	0.4990 	0.6529 	0.8742 	0.4876 
    MLP	0.6786 	0.7519 	0.4074 	0.6070 	0.8742 	0.4215 
    KNN	0.6107 	0.6437 	0.2425 	0.5026 	0.8491 	0.2975 
    GBDT	0.7786 	0.8670 	0.5826 	0.7534 	0.8805 	0.6446 
    Bagging	0.7250 	0.7928 	0.4676 	0.7064 	0.8050 	0.6198 
    AdaBoost	0.7393 	0.8143 	0.4913 	0.7283 	0.7925 	0.6694 
    MEDBN(ours)	0.7964 	0.8604 	0.6351 	0.7826 	0.7107 	0.8616 
    """,
    """
    RF	0.8450 	0.9150 	0.7199 	0.8327 	0.8878 	0.7810 
    ET	0.8041 	0.8614 	0.6250 	0.7631 	0.9171 	0.6350 
    XGBoost	0.8363 	0.9030 	0.6833 	0.8273 	0.8683 	0.7883 
    LogR	0.7573 	0.8482 	0.5473 	0.7396 	0.8146 	0.6715 
    SVM	0.6930 	0.7525 	0.3740 	0.6478 	0.8098 	0.5182 
    MLP	0.7076 	0.7423 	0.3984 	0.6912 	0.7610 	0.6277 
    KNN	0.6140 	0.6231 	0.1764 	0.5373 	0.7756 	0.3723 
    GBDT	0.8538 	0.9106 	0.7077 	0.8428 	0.8927 	0.7956 
    Bagging	0.7690 	0.8235 	0.5493 	0.7426 	0.8488 	0.6496 
    AdaBoost	0.8070 	0.8472 	0.5885 	0.7805 	0.8878 	0.6861 
    MEDBN(ours)	0.8567 	0.8897 	0.6957 	0.8401 	0.7737 	0.9122 
    """,
    """
    RF	0.8037 	0.8807 	0.6103 	0.7725 	0.8764 	0.6810 
    ET	0.7991 	0.8739 	0.5932 	0.7277 	0.9382 	0.5644 
    XGBoost	0.8105 	0.8882 	0.6316 	0.7932 	0.8545 	0.7362 
    LogR	0.7900 	0.8502 	0.5476 	0.7610 	0.8582 	0.6748 
    SVM	0.6849 	0.7488 	0.3948 	0.6610 	0.7418 	0.5890 
    MLP	0.6872 	0.7464 	0.3719 	0.6607 	0.7491 	0.5828 
    KNN	0.6164 	0.6135 	0.1560 	0.5394 	0.7527 	0.3865 
    GBDT	0.7991 	0.8782 	0.5928 	0.7831 	0.8400 	0.7301 
    Bagging	0.7580 	0.8292 	0.4996 	0.7491 	0.7818 	0.7178 
    AdaBoost	0.7854 	0.8503 	0.5453 	0.7631 	0.8400 	0.6933 
    MEDBN(ours)	0.8425 	0.9007 	0.7012 	0.8296 	0.7853 	0.8764 
    """,
    """
    RF	0.7560 	0.8603 	0.5857 	0.7586 	0.7904 	0.7282 
    ET	0.7694 	0.8549 	0.5522 	0.7712 	0.7904 	0.7524 
    XGBoost	0.7748 	0.8648 	0.5820 	0.7751 	0.7784 	0.7718 
    LogR	0.7131 	0.8062 	0.5082 	0.7043 	0.6467 	0.7670 
    SVM	0.6810 	0.7235 	0.3800 	0.6778 	0.6527 	0.7039 
    MLP	0.6729 	0.7269 	0.3498 	0.6602 	0.5868 	0.7427 
    KNN	0.5657 	0.6638 	0.2448 	0.5229 	0.8802 	0.3107 
    GBDT	0.7828 	0.8612 	0.5877 	0.7812 	0.7665 	0.7961 
    Bagging	0.7292 	0.7921 	0.4640 	0.7197 	0.6587 	0.7864 
    AdaBoost	0.7614 	0.8404 	0.5427 	0.7601 	0.7485 	0.7718 
    MEDBN(ours)	0.8391 	0.8991 	0.6904 	0.8342 	0.8738 	0.7964 
    """
]


def parse_results(results):
    data_lines = results.strip().split("\n")
    parsed_data = [line.split() for line in data_lines]
    models = [line[0] for line in parsed_data]
    acc = [float(line[1]) for line in parsed_data]
    auc = [float(line[2]) for line in parsed_data]
    ks = [float(line[3]) for line in parsed_data]
    g_mean = [float(line[4]) for line in parsed_data]
    return pd.DataFrame(
        {
            "Model": models,
            "Acc": acc,
            "AUC": auc,
            "KS": ks,
            "G-mean": g_mean,
        }
    )


def parse_df(all_results, metric, rank_data=False):
    n_dataset = len(all_results)
    all_res_df = []
    for res in all_results:
        res_df = parse_results(res)
        all_res_df.append(res_df)
    merged_res_df = pd.concat(all_res_df, axis=1, keys=[f'{i}' for i in range(n_dataset)])
    merged_res_df.columns = ['_'.join(col).strip() for col in merged_res_df.columns.values]
    metric_res_all_mdl_dt = merged_res_df[['0_Model'] + [f'{i}_{metric}' for i in range(n_dataset)]]
    if rank_data:
        # 对每一列进行排名
        rank_df = metric_res_all_mdl_dt.iloc[:, 1:].rank(method='min', ascending=False)

        # 将排名结果替换原 dataframe 中的准确率结果
        metric_res_all_mdl_dt.iloc[:, 1:] = rank_df
        # 对每一行求平均值
        metric_res_all_mdl_dt['Avg_Rank'] = metric_res_all_mdl_dt.iloc[:, 1:].mean(axis=1)
        print(metric_res_all_mdl_dt.iloc[:, 1:].values.tolist())
    return metric_res_all_mdl_dt

res_dicts = []
metrics = ['Acc', 'AUC', 'KS', 'G-mean']
for metric in metrics:
    res_df = parse_df(all_results, metric)
    # 移除 'Model' 列
    acc_columns = res_df.drop('0_Model', axis=1)
    # 进行Friedman检验
    stat, p = friedmanchisquare(*[acc_columns[col] for col in acc_columns])
    print(metric)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    res_dict = res_df.set_index('0_Model').T.to_dict('list')
    res_dicts.append(res_dict)

# nemenyi_plot_multiple(res_dicts, [f"({chr(ord('e') + i)}) {m}" for i, m in enumerate(metrics)], row=4, col=1)
nemenyi_plot_multiple(res_dicts, [f"({chr(ord('a') + i)}) {m}" for i, m in enumerate(metrics)], row=2, col=2)
