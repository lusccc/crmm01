import pandas as pd
import glob
from pathlib import Path
from datetime import datetime

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

sec_res_txt = glob.glob('sec_res/**/**/*.txt', recursive=True)  # => ['2.txt', 'sub/3.txt']

# corp_rating_df = pd.read_csv('corporate_rating_with_cik.csv', index_col=0)
corp_rating_df = pd.read_csv('../../data/cr_sec_ori/corporate_rating_with_cik.csv', index_col=0)
corp_rating_df['Item7A'] = ''
corp_rating_df['Item7'] = ''
corp_rating_df['Item2'] = ''
corp_rating_df['Item1A'] = ''
corp_rating_df['Item1'] = ''

for txt_file in sec_res_txt:
    fname = Path(txt_file).name
    fname_split = fname.split('_')

    # sec info and company info
    symbol, cik, sec_type, sec_report_date, item_type = \
        fname_split[0], fname_split[1], fname_split[2], fname_split[3], fname_split[4]
    sec_report_date = datetime.strptime(sec_report_date, '%Y%m%d')
    print(symbol, cik, sec_type, sec_report_date, item_type)
    if item_type in ['Item7A', 'Item7', 'Item2', 'Item1A', 'Item1']:
        corp_rating_rec_df = corp_rating_df.loc[corp_rating_df['Symbol'] == symbol]
        for idx, row in corp_rating_rec_df.iterrows():
            rating_date = datetime.strptime(row['Date'], '%m/%d/%Y')
            query_start_date = rating_date - relativedelta(years=1)
            if query_start_date <= sec_report_date <= rating_date:
                # remove line break and ignore unicode char
                txt_content = Path(txt_file).read_text().replace('\n', '').encode('ascii', 'ignore').decode()
                corp_rating_df.at[idx, item_type] = txt_content

corp_rating_df.to_csv('corporate_rating_with_cik_and_sec_text.csv')
