import os
import re
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

from sec_edgar_text.download import EdgarCrawler
from sec_edgar_text.utils import logger, storage_toplevel_directory, args, date_search_string

MAX_FILES_IN_SUBDIRECTORY = 1000


def main():
    # dataset_name = 'cr'
    dataset_name = 'cr2'
    if dataset_name == 'cr':
        query_data = pd.read_csv('./data/cr_sec_ori/corp_rating_sec_df_na.csv')[['Symbol', 'Date', 'CIK']]
    elif dataset_name == 'cr2':
        query_data = pd.read_csv('./data/cr2_sec_ori/corporateCreditRatingWithFinancialRatios.csv')[
            ['Ticker', 'Rating Date', 'CIK']]
    else:
        raise ValueError('dataset name not supported')

    companies = query_data.loc[:]
    storage_subdirectory_number = 1
    for c, company_keys in companies.iterrows():

        print(f'!!!!!!!!!!{company_keys}')
        # below are respectively 'Symbol', 'Date', 'CIK'
        company_description, rating_date, edgar_search_string = list([*company_keys])
        company_description = company_description.strip()
        company_description = re.sub('/', '', company_description)

        if dataset_name == 'cr':
            dt_format = '%m/%d/%Y'
        elif dataset_name == 'cr2':
            dt_format = '%Y-%m-%d'
        rating_date = datetime.strptime(rating_date, dt_format)
        query_start_date = rating_date - relativedelta(years=1)

        logger.info(' begin downloading company: ' +
                    str(c + 1) + ' / ' +
                    str(len(companies)))
        storage_subdirectory = os.path.join(storage_toplevel_directory,
                                            format(storage_subdirectory_number,
                                                   '03d'))
        if not os.path.exists(storage_subdirectory):
            os.makedirs(storage_subdirectory)

        seccrawler = EdgarCrawler()
        seccrawler.storage_folder = storage_subdirectory
        filings_metadata = []
        for filing_search_string in args.filings:
            metadata = seccrawler.download_filings(company_description,
                                                   edgar_search_string,
                                                   filing_search_string,
                                                   date_search_string,
                                                   query_start_date.strftime('%Y%m%d'),
                                                   rating_date.strftime('%Y%m%d'),
                                                   False)
            filings_metadata.append(metadata)
        if len(os.listdir(storage_subdirectory)) > MAX_FILES_IN_SUBDIRECTORY:
            storage_subdirectory_number += 1

        # break


if __name__ == '__main__':
    main()
