import itertools
from multiprocessing import Lock, Pool

import pandas as pd
from tqdm import tqdm

from summa_score_sentences import summarize

bert_word_limit = 512

"""
text rank extract most important sentence, but not work in multi modal dbn!
"""
def extract_important_sent(txts, pno):
    res = []
    for row_id, row in txts.iterrows():
        print(f'pno:{pno}, row id: {row_id}')
        txt = row['secText']
        # sentence importance form high to low
        rank = summarize(txt)
        total_word_num = 0
        extracted_sent = []
        for i, r in enumerate(rank[0]):  # each sentence and score
            sentence = r.text
            word_num = len(sentence.split())
            total_word_num += word_num
            if total_word_num <= bert_word_limit or len(extracted_sent) == 0:
                extracted_sent.append(sentence)
                break#!!!!!!!
            else:
                break
        res.append([row_id, ''.join(extracted_sent)])
    return res


def init(l):
    global lock
    lock = l


if __name__ == '__main__':
    data_df = pd.read_csv('../../data/cr_sec_ori/corporate_rating_with_cik_and_sec_merged_text.csv', index_col=0)
    # for index, row in data_df.iterrows():
    #     txt = row['secText']

    processes = 24
    lock = Lock()

    num = len(data_df)
    multi_thread_tasks = []
    process_pool = Pool(processes=processes, initializer=init, initargs=(lock,))
    task_batch_size = int(num / processes + 1)
    multi_thread_tasks = []
    pno = 0

    for t in range(0, num, task_batch_size):
        print(f'{t}:{t + task_batch_size}')
        batch_tasks = data_df.loc[t:t + task_batch_size - 1]  # `loc` is Closed interval, hence -1
        multi_thread_tasks.append(process_pool.apply_async(extract_important_sent,
                                                           (batch_tasks, pno),
                                                           # callback=lambda x: pbar.update(task_batch_size),
                                                           ))
        pno += 1

    exec_res = [t.get() for t in multi_thread_tasks]
    res = list(itertools.chain(*exec_res))
    process_pool.close()  # 关闭线程池清理

    for ranked in res:
        row_id, txt = ranked
        data_df.at[row_id, 'secText'] = txt

    data_df.to_csv('../../data/cr_sec_ori/corporate_rating_with_cik_and_ranked_sec_text.csv')
