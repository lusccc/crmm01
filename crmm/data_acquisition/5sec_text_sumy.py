import itertools
from multiprocessing import Lock, Pool

import pandas as pd
from keybert import KeyBERT
from tqdm import tqdm

# from summa_score_sentences import summarize

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# https://github.com/miso-belica/sumy
# https://blog.csdn.net/Together_CZ/article/details/107628462

LANGUAGE = "english"
# SENTENCES_COUNT = 2
SENTENCES_COUNT = 10

"""
make sure you have aleady done below:
import nltk
nltk.download('punkt')
"""


def extract_important_sent(txts, pno):
    res = []
    for row_id, row in txts.iterrows():
        print(f'pno:{pno}, row id: {row_id}')
        txt = row['secText']
        parser = PlaintextParser.from_string(txt, Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)
        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)
        sum_sent = []
        for sentence in summarizer(parser.document, SENTENCES_COUNT):
            sum_sent.append(str(sentence))
        summed = ''.join(sum_sent)

        res.append([row_id, summed])
    return res


def init(l):
    global lock
    lock = l


if __name__ == '__main__':
    data_df = pd.read_csv('../../data/cr_sec_ori/corporate_rating_with_cik_and_sec_merged_text.csv', index_col=0)
    # for index, row in data_df.iterrows():
    #     txt = row['secText']

    processes = 10
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

    data_df.to_csv('../../data/cr_sec_ori/corporate_rating_with_cik_and_summarized_sec_text.csv')
