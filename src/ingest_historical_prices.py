#!/usr/bin/env python3
"""
Responsible for loading historical ASX prices and converting them into django model equivalents.
Not all data can be provided: fields which are missing are set to None. Historical prices are loaded
from google finance API using pandas_data_reader. This script is carefully written to not alter 
data already existing in the specified database.
"""
import argparse
import os
import re
import zipfile
from datetime import date
import pandas as pd
from io import TextIOWrapper
import csv
import pymongo
from collections import namedtuple
from utils import read_config
from datetime import datetime

def prune_existing_records(db, records_to_be_saved):
    """
    Return a list of records which are not already present in the db from the input list
    """
    ok_records = []
    fetch_dates = set([rec['fetch_date'] for rec in records_to_be_saved])
    pre_existing = set()
    for fd in fetch_dates:
        stocks = db.asx_prices.distinct('asx_code', {'fetch_date': fd})
        for stock in stocks:
            pre_existing.add("{}-{}".format(stock, fd))
    for rec in records_to_be_saved:
        key = "{}-{}".format(rec['asx_code'], rec['fetch_date'])
        if key not in pre_existing:
            ok_records.append(rec)
    return ok_records

def save_batch(db, named_tuples, known_descriptions):
    assert db is not None

    print("Saving data batch to db with {} records".format(len(named_tuples)))
    records = []
    for nt in named_tuples:
        rec = dict(nt._asdict())
        #print(nt)
        # NB: need to compute as many fields of the Quotation model as possible to help the viewer app
        rec.update({ 'error_code': '', 'error_descr': ''})
        rec['fetch_date'] = nt.fetch_date.strftime("%Y-%m-%d")
        assert len(rec['fetch_date']) == 10 
        assert re.match(r'^\d{4}-\d{2}-\d{2}', rec['fetch_date'])
        assert 'asx_code' in rec
        change_price = nt.close_price - nt.open_price
        cip = change_price / nt.open_price * 100.0 if nt.open_price > 0.0 else 0.0
        rec.update({'volume': nt.volume, 
                    'last_price': nt.close_price, 
                    'average_daily_volume': nt.volume,
                    'code': nt.asx_code,
                    'last_trade_date': nt.fetch_date,
                    'day_high_price': nt.high_price,
                    'day_low_price': nt.low_price,
                    'offer_price': nt.open_price, 
                    'change_price': change_price,
                    'change_in_percent': cip,
                    'open_price': nt.open_price,
                    'bid_price': nt.open_price,
        })
        rec.update({'suspended':False}) # no way to determine this based on available data, so just assume false

        # no way to compute this from available data as its being processed - so set to None for now
        rec.update({'annual_dividend_yield': None, 
                    'deprecated_market_cap': None, 
                    'deprecated_number_of_shares': None,
                    'previous_close_price': None,
                    'previous_day_percentage_change': None,
                    'eps': float('nan'),  # flag as missing data for django app
                    'market_cap': None,
                    'number_of_shares': None,
                    'pe': float('nan'),
                    'year_high_date': None,
                    'year_high_price': float('nan'),
                    'year_low_price': float('nan'),
                    'year_low_date': None})

        # use existing database if possible for these, otherwise default
        known_descr, known_isin = known_descriptions.get(nt.asx_code, (None, None))
        rec.update({'descr_full': known_descr, "isin_code": known_isin})
        records.append(rec)

    records = prune_existing_records(db, records)
    if len(records) > 0:
        try:
            result = db.asx_prices.insert_many(records)
            assert result is not None
            assert isinstance(result, pymongo.results.InsertManyResult)
            assert result.acknowledged
            assert len(records) == len(result.inserted_ids) # every record must be inserted
        except pymongo.errors.BulkWriteError as bwe:
            print(bwe.details)
            raise

def process_eod_data(filename, csv_file, db, known_descriptions, keep_old_stocks=False):
    """
    Processing the specified filename with the specified content. Wont update the database if the 
    date of the entry corresponds to data in the database.
    """
    date_portion = filename[0:6]
    n = skipped = 0
    batch = []
    quote = namedtuple('Quote', ['volume', 'asx_code', 'open_price', 'close_price', 'high_price', 'low_price', 'fetch_date'])
    for row in csv_file:
        n += 1
        # csv rows must be: Ticker, Date, Open, High, Low, Close, Volume
        # we ignore data which doesnt look right to avoid polluting the db
        try:
            assert len(row) == 7
            asx_code, trading_date, open_price, high_price, low_price, close_price, volume = row
            assert trading_date.startswith(date_portion)
            asx_code = asx_code.rstrip('.') # data feed sometimes has these, accept if the other fields are ok
            assert re.match(r'^\w+$', asx_code)
            open_price = float(open_price)
            high_price = float(high_price)
            low_price = float(low_price)
            close_price = float(close_price)
            volume = int(volume)
            #print(low_price, " ", close_price, " ", high_price, " ", row)
            assert low_price <= high_price
            assert close_price >= low_price and close_price <= high_price
            # if we get here it is a valid quote so we add it to the batch
            if keep_old_stocks or asx_code in known_descriptions:
                batch.append(quote(asx_code=asx_code,
                                fetch_date=datetime.strptime(trading_date, "%Y%m%d"),
                                open_price=open_price,
                                close_price=close_price,
                                high_price=high_price,
                                low_price=low_price,
                                volume=volume))
        except AssertionError as ae:
            #print(f"Rejecting invalid end-of-day data: {row}")
            skipped += 1
        except ValueError:
            #print(f"Rejecting badly formatted data: {row}")
            skipped += 1
    if skipped > 0:
        print(f"Skipped {skipped} records (total {n}) in {filename}")
    save_batch(db, batch, known_descriptions)
    return (n, skipped)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Load downloaded zip files eg. from asxhistoricalprices.com into MongoDB")
    args.add_argument('--dir', help="Directory to scan for end-of-day zip files", required=True, type=str)
    args.add_argument('--config', help="Configuration file to use [config.json]", type=str, default="config.json")
    args.add_argument('--keep-delisted', help="Retain delisted stocks in DB", action='store_true')
    a = args.parse_args()
    config, password = read_config(a.config)
    m = config.get('mongo')
    mongo = pymongo.MongoClient(m.get('host'), m.get('port'), username=m.get('user'), password=password)
    db = mongo[m.get('db')]

    known_stocks = set(db.asx_prices.distinct('asx_code'))
    known_descriptions = {}
    for stock in known_stocks:
        rec = db.asx_prices.find_one({'asx_code': stock, 'descr_full': {'$exists': True }})
        #print(rec)
        if rec is not None and len(rec['descr_full']) > 0:
            known_descriptions[stock] = (rec['descr_full'], rec['isin_code'])
    print("Identified {} known stock descriptions.".format(len(known_descriptions)))

    files_to_process = []
    for dirpath, subdirs, filenames in os.walk(a.dir):
        zip_files = [os.path.join(dirpath, filename) for filename in filenames 
                     if filename.lower().endswith(".zip")]
        files_to_process.extend(zip_files)
    print("Found {} zip files to scan for end-of-day data".format(len(files_to_process)))
    sum_skipped = sum_n = skipped_files = 0
    for zf in zip_files:
        with zipfile.ZipFile(zf, 'r') as zip_file:
            entries = zip_file.namelist()
            print("*** processing {}... has {} files to process".format(zf, len(entries)))
            for entry in entries:
                if 'MACOS' in entry:
                    continue
                m = re.match(r'^(.*/)?(\d{6,8}.txt)$', entry)
                if not m:
                    print(f"*** WARNING: skipping entry without valid filename: {entry}")
                    skipped_files += 1
                    continue

                with zip_file.open(entry, 'r') as fp:
                    total_n, total_skipped = process_eod_data(m.group(2),
                                                              csv.reader(TextIOWrapper(fp), dialect='excel', delimiter=','), 
                                                              db,
                                                              known_descriptions,
                                                              keep_old_stocks=a.keep_delisted)
                    assert total_skipped < total_n
                    sum_skipped += total_skipped
                    sum_n += total_n
    print(f"Processed {sum_n} records, skipped {sum_skipped}")
    print(f"Rejected {skipped_files} files as no recognisable filename found")

exit(0)
