import re
import pandas as pd
from nltk import tokenize
import warnings
import itertools
from multiprocessing import Pool
# import nltk
# nltk.download('punkt')
warnings.filterwarnings('ignore')

data_df = pd.read_csv('../../data/cr_sec_ori/corporate_rating_with_cik_and_sec_merged_text.csv')
split_sec_data_df = pd.DataFrame(columns=data_df.columns)

def split_sec_to_sent_create_new_df(df):
    rows = []
    for index, row in df.iterrows():
        except_txt = row[:-1]
        sec_txt = row['secText']
        # sentences = split_into_sentences(sec_txt)
        sentences = tokenize.sent_tokenize(sec_txt)
        for sent in sentences:
            new_row = dict(except_txt)
            new_row['secText'] = sent
            # split_sec_data_df = pd.concat([split_sec_data_df, except_txt], ignore_index=True)
            # split_sec_data_df = split_sec_data_df.append(except_txt, ignore_index=True)
            rows.append(new_row)
    return rows



processes = 12

num_row = len(data_df)
print(f'num_row: {num_row}')
multi_thread_tasks = []
process_pool = Pool(processes=processes,)
task_batch_size = int(num_row / processes + 1)
multi_thread_tasks = []
total = 0
for t in range(0, num_row, task_batch_size):
    batch_tasks = data_df.loc[t:t + task_batch_size]
    multi_thread_tasks.append(process_pool.apply_async(split_sec_to_sent_create_new_df,
                                                       (batch_tasks,),
                                                       # callback=lambda x: pbar.update(task_batch_size),
                                                       ))
    total += len(batch_tasks)
print(f'total: {total}')

exec_res = [t.get() for t in multi_thread_tasks]
res = list(itertools.chain(*exec_res))
process_pool.close()  # 关闭线程池清理
split_sec_data_df = pd.DataFrame(res)
split_sec_data_df.columns = data_df.columns
print(f'after split: {len(split_sec_data_df)}')
split_sec_data_df.to_csv('../../data/cr_sec_ori/corporate_rating_with_cik_and_sec_sentences.csv')