import glob
import itertools
import re
from datetime import datetime
from multiprocessing import Pool, Lock
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

"""
MD&A:
Item 7A. Quantitative and Qualitative Disclosure About Market Risk
Item 7. Managements Discussion and Analysis of Financial Condition & Results of Operations
Item 2. Managements Discussion and Analysis of Financial Condition and Results of Operations
Item 1A. Risk Factors
Item 1. Financial Statements:
# Item 2. Properties
"""


# https://stackoverflow.com/questions/27484660/removing-pattern-matched-line-from-multiline-string-in-python
def remove_line_contain(rem, my_string):
    return re.sub(".*" + rem + ".*\n?", "", my_string)


def remove_only_digit_line(my_string):
    # in sec text, there exists digit line, in fact it is the page number. e.g., `38`
    # note space may after number
    """
    (?m) declares the regex to read multiline data.
    ^ declares the start of line.
    \d declares a digit.
    \d+ declares that there should be at least 1 digit.
    (\d+) declares a group so that you could extract the numbers from input lines.
    . declares any character.
    .* declares that there may be any number of any characters, that means any line (even an empty one).
    """
    s1 = re.sub('(?m)^\d+ +$', "", my_string)  # space after number
    s2 = re.sub('(?m)^\d+$', "", s1)  # no space
    s3 = re.sub('(?m)^ +\d+$', "", s2)  # space before number
    return s3


def process_sec_text(txts, corp_rating_df, symbol_col, date_col, date_format):
    loc_item_txt = []
    for txt_file in txts:
        fname = Path(txt_file).name
        fname_split = fname.split('_')

        # sec info and company info
        symbol, cik, sec_type, sec_report_date, item_type = \
            fname_split[0], fname_split[1], fname_split[2], fname_split[3], fname_split[4]
        sec_report_date = datetime.strptime(sec_report_date, '%Y%m%d')
        # print(symbol, cik, sec_type, sec_report_date, item_type)
        if item_type in use_sec_section:
            corp_rating_rec_df = corp_rating_df.loc[corp_rating_df[symbol_col] == symbol]
            for idx, row in corp_rating_rec_df.iterrows():
                rating_date = datetime.strptime(row[date_col], date_format)
                query_start_date = rating_date - relativedelta(years=1)
                if query_start_date <= sec_report_date <= rating_date:
                    # ignore unicode char
                    # txt_content = Path(txt_file).read_text().replace('\n', '').encode('ascii', 'ignore').decode()
                    txt_content = Path(txt_file).read_text().encode('ascii', 'ignore').decode()
                    txt_content_rem = remove_line_contain('ITEM', txt_content)
                    txt_content_rem1 = remove_line_contain('Table of Contents', txt_content_rem)
                    txt_content_rem2 = remove_only_digit_line(txt_content_rem1)
                    # lock.acquire()
                    print(f'write idx: {idx}, item: {item_type}')
                    # corp_rating_df.at[idx, item_type] = txt_content_rem2
                    loc_item_txt.append([idx, item_type, txt_content_rem2])
                    # lock.release()

    return loc_item_txt


def init(l):
    global lock
    lock = l


# use_sec_section = ['Item7A', 'Item7', 'Item2', 'Item1A', 'Item1']
use_sec_section = ['Item2']  # TODO make comparison

def main():

    dataset_name = 'cr2'
    if dataset_name == 'cr':
        sec_dir_name = 'cr_sec_crawler_res'
        ori_data_path = './data/cr_sec_ori/corporate_rating_with_cik.csv'
        symbol_col = 'Symbol'
        date_col = 'Date'
        date_format = '%m/%d/%Y'
    elif dataset_name == 'cr2':
        sec_dir_name = 'cr2_sec_crawler_res'
        ori_data_path = './data/cr2_sec_ori/corporateCreditRatingWithFinancialRatios.csv'
        symbol_col = 'Ticker'
        date_col = 'Rating Date'
        date_format = '%Y-%m-%d'
    else:
        raise ValueError(f'invalid dataset name: {dataset_name}')

    corp_rating_df = pd.read_csv(ori_data_path)
    # we should assign Id as index to locate sec text with csv dataframe
    if 'Id' not in corp_rating_df.columns:
        corp_rating_df['Id'] = corp_rating_df.index
    corp_rating_df.set_index('Id', inplace=True)

    for sec_type in use_sec_section:
        corp_rating_df[sec_type] = ''

    processes = 20
    lock = Lock()

    sec_res_txt = glob.glob(f'./{sec_dir_name}/**/**/*.txt', recursive=True)  # => ['2.txt', 'sub/3.txt']
    txt_file_num = len(sec_res_txt)
    multi_thread_tasks = []
    process_pool = Pool(processes=processes, initializer=init, initargs=(lock,))
    task_batch_size = int(txt_file_num / processes + 1)
    multi_thread_tasks = []
    pno = 0

    for t in range(0, txt_file_num, task_batch_size):
        batch_tasks_txt = sec_res_txt[t:t + task_batch_size]
        multi_thread_tasks.append(process_pool.apply_async(process_sec_text,
                                                           (batch_tasks_txt, corp_rating_df, symbol_col, date_col,
                                                            date_format),
                                                           # callback=lambda x: pbar.update(task_batch_size),
                                                           ))
        pno += 1

    exec_res = [t.get() for t in multi_thread_tasks]
    res = list(itertools.chain(*exec_res))
    process_pool.close()

    for loc_item_txt in res:
        loc, item, txt = loc_item_txt
        corp_rating_df.at[loc, item] = txt

    corp_rating_df.to_csv(f'./data/{dataset_name}_sec_ori/corporate_rating_with_cik_and_sec_text.csv')


if '__main__' == __name__:
    main()
