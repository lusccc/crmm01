import pandas as pd
import numpy as np

n_class = 6

data_df = pd.read_csv('data/cr_sec_ori/corporate_rating_with_cik_and_sec_merged_text.csv')

if n_class == 10:
    dataset_name = 'cr_sec'
    data_df = data_df.replace(
        {'Rating': {'AAA': 0, 'AA': 1, 'A': 2, 'BBB': 3, 'BB': 4, 'B': 5, 'CCC': 6, 'CC': 7, 'C': 8, 'D': 9}}
    )
    data_labels = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC']
    data_labels = np.array(data_labels)
elif n_class == 6:
    dataset_name = 'cr_sec_6'
    data_df = data_df.replace(
        {'Rating': {'AAA': 0, 'AA': 0, 'A': 1, 'BBB': 2, 'BB': 3, 'B': 4, 'CCC': 5, 'CC': 5, 'C': 5, 'D': 5}}
    )
    data_labels = ['AA+', 'A', 'BBB', 'BB', 'B', 'CCC-']
    data_labels = np.array(data_labels)

seed = 3407  # !!!!!! USED TO RANDOM SHUFFLE
cr_sec_all_df = data_df.sample(frac=1, random_state=seed)

n_sample = len(cr_sec_all_df)
train_split_idx = int(n_sample * .7)
# 406 is the test set size in paper `Multimodal Machine Learning for Credit Modeling`!
val_split_idx = int(n_sample * (1 - 406 / n_sample))
train_df, val_df, test_df = np.split(cr_sec_all_df, [train_split_idx, val_split_idx])
print('Num examples train-val-test')
print(len(train_df), len(val_df), len(test_df))

cr_sec_all_df.to_csv(f'data/{dataset_name}/all.csv')
train_df.to_csv(f'data/{dataset_name}/train.csv')
val_df.to_csv(f'data/{dataset_name}/val.csv')
test_df.to_csv(f'data/{dataset_name}/test.csv')
np.save(f'data/{dataset_name}/label_list.npy', data_labels)
