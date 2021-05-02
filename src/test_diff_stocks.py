#!/usr/bin/python3.8
import argparse
from typing import Iterable
import pymongo
import pandas as pd
from utils import read_config, now

def stocks_by_sector(asx_company_details) -> pd.DataFrame:
    rows = [
        (d.get('asx_code'), d.get('sector_name'))  for d in asx_company_details.find()
    ]
    df = pd.DataFrame.from_records(rows, columns=['asx_code', 'sector_name'])
    print(df)
    assert len(df) > 0
    colnames = set(df.columns)
    return df

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Load Australian Bureau of Statistics (ABS) datasets and create matching inverted index for abs app to use")
    args.add_argument('--config', help="Configuration file to use [config.json]", type=str, default="config.json")
    a = args.parse_args()
    config, password = read_config(a.config)
    m = config.get('mongo')
    mongo = pymongo.MongoClient(m.get('host'), m.get('port'), username=m.get('user'), password=password)
    db = mongo[m.get('db')]

    current_stocks = set(db.asx_prices.distinct('asx_code', { 'fetch_date': '2021-04-28' }))
    prior_stocks = set(db.asx_prices.distinct('asx_code', { 'fetch_date': '2021-04-01' }))

    s1 = current_stocks.difference(prior_stocks)
    print("New stocks from ASX ISIN change: {}".format(s1))
    s2 = prior_stocks.difference(current_stocks)
    print("Stocks lost from ASX ISIN change: {}".format(s2))
    ss = stocks_by_sector(db.asx_company_details)
    ss = ss.set_index('asx_code')
    #print(ss)
    ss = ss.filter(s1, axis=0)
    print(ss)
    exit(0)
