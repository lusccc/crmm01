import argparse
import itertools
from math import isnan

import pandas as pd
import torch
from keybert import KeyBERT


def extract_sec_keywords(txts, keywords_num, pno):
    res = []
    kw_model = KeyBERT()
    for row_id, row in txts.iterrows():
        print(f'pno:{pno}, row id: {row_id}')
        txt = row['secText']
        if txt == 'nan' or (isinstance(txt, float) and isnan(txt)):
            res.append([row_id, ''])
            continue
        try:
            keywords = kw_model.extract_keywords(txt, top_n=int(keywords_num))  # (word, score) tuples
        except:
            print()
        keywords = ' '.join([w[0] for w in keywords])
        res.append([row_id, keywords])
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prev_step_sentences_num', type=int, default=10)
    parser.add_argument('--keywords_num', type=int, default=30)
    parser.add_argument('--dataset_name', type=str, default='cr2')
    args = parser.parse_args()

    sentences_num = args.prev_step_sentences_num
    keywords_num = args.keywords_num
    dataset_name = args.dataset_name

    torch.multiprocessing.set_start_method('spawn')
    data_df = pd.read_csv(
        f'./data/{dataset_name}_sec_ori/corporate_rating_with_cik_and_summarized_sec_sent{sentences_num}.csv',
        index_col=0)

    processes = 3
    ctx = torch.multiprocessing.get_context("spawn")
    process_pool = ctx.Pool(processes=processes, )

    num = len(data_df)
    multi_thread_tasks = []
    task_batch_size = int(num / processes + 1)
    multi_thread_tasks = []
    pno = 0

    for t in range(0, num, task_batch_size):
        print(f'{t}:{t + task_batch_size}')
        batch_tasks = data_df.loc[t:t + task_batch_size - 1]  # `loc` is Closed interval, hence -1
        multi_thread_tasks.append(process_pool.apply_async(extract_sec_keywords,
                                                           (batch_tasks, keywords_num, pno),
                                                           # callback=lambda x: pbar.update(task_batch_size),
                                                           ))
        pno += 1

    exec_res = [t.get() for t in multi_thread_tasks]
    res = list(itertools.chain(*exec_res))
    process_pool.close()  # 关闭线程池清理

    data_df.insert(data_df.shape[1], 'secKeywords', '')
    for ranked in res:
        row_id, keywords = ranked
        data_df.at[row_id, 'secKeywords'] = keywords

    data_df.to_csv(
        f'./data/{dataset_name}_sec_ori/corporate_rating_with_cik_and_summarized_sec_sent{sentences_num}_keywords{keywords_num}.csv')
