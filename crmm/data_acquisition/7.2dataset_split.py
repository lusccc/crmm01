import argparse
import os

import numpy as np
import pandas as pd


def process_dataset(data_df, n_class, split_method, train_years, test_years, dataset_name):
    # @@@@  1. mapping labels
    if n_class:  # manually define class number
        if dataset_name == 'cr':
            if n_class == 10:
                data_df = data_df.replace(
                    {'Rating': {'AAA': 0, 'AA': 1, 'A': 2, 'BBB': 3, 'BB': 4, 'B': 5, 'CCC': 6, 'CC': 7, 'C': 8,
                                'D': 9}}
                )
                data_labels = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC']
            elif n_class == 6:
                data_df = data_df.replace(
                    {'Rating': {'AAA': 0, 'AA': 0, 'A': 1, 'BBB': 2, 'BB': 3, 'B': 4, 'CCC': 5, 'CC': 5, 'C': 5,
                                'D': 5}}
                )
                data_labels = ['AA+', 'A', 'BBB', 'BB', 'B', 'CCC-']
            elif n_class == 2:
                data_df = data_df.replace(
                    {'Rating': {'AAA': 1, 'AA': 1, 'A': 1, 'BBB': 1, 'BB': 0, 'B': 0, 'CCC': 0, 'CC': 0, 'C': 0,
                                'D': 0}}
                )
                data_labels = ['good', 'junk']
        elif dataset_name == 'cr2':
            if n_class == 6:
                data_df = data_df.replace(
                    {'Rating': {'AAA': 0, 'AA+': 0, 'AA': 0, 'AA-': 0,
                                'A+': 1, 'A': 1, 'A-': 1,
                                'BBB+': 2, 'BBB': 2, 'BBB-': 2,
                                'BB+': 3, 'BB': 3, 'BB-': 3,
                                'B+': 4, 'B': 4, 'B-': 4,
                                'CCC+': 5, 'CCC': 5, 'CCC-': 5,
                                'CC+': 5, 'CC': 5, 'C': 5,
                                'D': 5}}
                )
                data_labels = ['AA+', 'A', 'BBB', 'BB', 'B', 'CCC-']
            elif n_class == 2:
                data_df.loc[:, 'Rating'] = data_df.loc[:, 'Binary Rating']
                data_labels = ['good', 'junk']
    else:  # use class provided by dataset
        data_labels = data_df['Rating'].unique()
        data_df['Rating'] = pd.factorize(data_df['Rating'])[0]

    print()

    # @@@@  2. split train-val-test
    data_df = data_df.sample(frac=1)
    if split_method == 'mixed':
        n_sample = len(data_df)
        train_split_idx = int(n_sample * .7)
        val_split_idx = int(n_sample * .8)
        # # 406 is the test set size in paper `Multimodal Machine Learning for Credit Modeling`!
        # val_split_idx = int(n_sample * (1 - 406 / n_sample))
        train_df, val_df, test_df = np.split(data_df, [train_split_idx, val_split_idx])

    elif split_method == 'by_year':
        print('-'*60)
        print(f'train_years: {train_years}, test_years: {test_years}')
        date_col = 'Rating Date' if 'Rating Date' in data_df.columns else 'Date'
        data_df[date_col] = pd.to_datetime(data_df[date_col])
        data_df['Rating Year'] = data_df[date_col].dt.year.astype(int)
        train_df = data_df[data_df['Rating Year'].isin(train_years)]
        train_df, val_df = np.split(train_df, [int(len(train_df) * .9)])
        test_df = data_df[data_df['Rating Year'].isin(test_years)]
    else:
        raise ValueError(f'Unknown split method: {split_method}')

    print(f'train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}')
    print('-' * 60)

    # @@@@  3. save
    if split_method == 'mixed':
        data_dir = f'./data/{dataset_name}_cls{n_class}_{split_method}_st{sentences_num}_kw{keywords_num}'
    else:
        data_dir = f'./data/{dataset_name}_cls{n_class}_{split_method}_st{sentences_num}_kw{keywords_num}' \
                   f'_{",".join(map(str, train_years))}_{"".join(map(str, test_years))}'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    train_df.to_csv(f'{data_dir}/train.csv')
    val_df.to_csv(f'{data_dir}/val.csv')
    test_df.to_csv(f'{data_dir}/test.csv')
    np.save(f'{data_dir}/label_list.npy', np.array(data_labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cr2', help='dataset name')
    parser.add_argument('--prev_step_sentences_num', type=int, default=10)
    parser.add_argument('--prev_step_keywords_num', type=int, default=20)
    parser.add_argument('--n_class', type=int, default=2, help='number of classes')
    parser.add_argument('--split_method', type=str, default='mixed', help='split method')
    parser.add_argument('--train_years', type=str, default='', help='train years')
    parser.add_argument('--test_years', type=str, default='', help='test years')
    args = parser.parse_args()
    print(args)

    train_years = [int(year) for year in args.train_years.split(',')]
    test_years = [int(year) for year in args.test_years.split(',')]

    dataset_name = args.dataset_name
    sentences_num = args.prev_step_sentences_num
    keywords_num = args.prev_step_keywords_num
    n_class = args.n_class
    split_method = args.split_method
    print(f'./data/{dataset_name}_sec_ori/'
          f'corporate_rating_with_cik_and_summarized_sec_sent{sentences_num}_keywords{keywords_num}.csv')
    data_df = pd.read_csv(f'./data/{dataset_name}_sec_ori/'
                          f'corporate_rating_with_cik_and_summarized_sec_sent{sentences_num}_keywords{keywords_num}.csv',
                          index_col=0)
    process_dataset(data_df, n_class, split_method, train_years, test_years, dataset_name=dataset_name)

# parameters set in data_acquisition/5sec_text_sumy.py and data_acquisition/6sec_keywords_extraction.py
# sentence_num = 2
# keywords_num = 20
#
# datasets = ['cr2']
# n_classes = [2, 6]
# split_methods = ['mixed', 'by_year']
# start_year = 2010
# end_year = 2016

# for dataset_name in datasets:
#     for n_class in n_classes:
#         for split_method in split_methods:
#             if split_method == 'mixed':
#                 process_dataset(n_class, split_method, train_years=[], test_years=[], dataset_name=dataset_name)
#             else:
#                 for year in range(start_year, end_year):
#                     train_years = list(range(start_year, year + 1))
#                     test_years = [year + 1]
#                     process_dataset(n_class, split_method, train_years, test_years, dataset_name=dataset_name)
