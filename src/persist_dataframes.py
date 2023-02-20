#!/usr/bin/python3
from builtins import exit, float, int, isinstance, len, list, print, range, set, str
import argparse
import io
import os
from datetime import datetime, date
import calendar
import hashlib
import pymongo
#from pyarrow import Table
#import pyarrow.parquet as pq
from bson.binary import Binary
import pandas as pd


def dates_of_month(month, year):
    assert month >= 1 and month <= 12
    assert year >= 1995
    num_days = calendar.monthrange(year, month)[1]
    days = [str(date(year, month, day)) for day in range(1, num_days+1)]
    #print(month, year, " ", days)
    return days

def clean_value(value):
    """
    To ensure the computed dataframe is consistently typed all values are
    co-erced to float or we die trying...
    """
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    # else str...
    value = value.rstrip('%')
    value = value.replace(',', '')
    return float(value)

def load_field_dataframe(db, field_name, month, year) -> list:
    """
    Build a matrix of all available stocks with the given field_name eg. 'last_price'. This
    is built into a pandas dataframe and returned. NaN's may be returned where the stock has no quotation at a given date.
    All dates in the specified month are included in the dataframe.
    The dataframe is always organised as stock X dates (stocks are rows) with the values of the specified field as float values
    """
    assert db is not None
    assert len(field_name) > 0
    days_of_month = dates_of_month(month, year)
    rows = [{ 'asx_code': row['asx_code'],
              'fetch_date': row['fetch_date'],
              'field_name': field_name,
              'field_value': clean_value(row[field_name])}
              for row in db.asx_prices.find({'fetch_date': { "$in": days_of_month},
                                              field_name: { "$exists": True },
                                             'suspended': { "$ne": True },
                                            },
                                            {'asx_code': 1, field_name: 1, 'fetch_date': 1})
           ]
   
    return rows

def save_dataframe(db, df, tag, field_name, n_days=None, n_stocks=None, **kwargs):
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    assert db is not None
    for colname in ['asx_code', 'fetch_date', 'field_name']:
        if colname in df.columns:
           print(f"Making {colname} a categorical column")
           df[colname] = pd.Categorical(df[colname], categories=df[colname].unique())
    df['field_value'] = pd.to_numeric(df['field_value'], downcast='float')
    df = df.sort_values(['field_name', 'field_value'], axis=0) # sorting improves parquet compression and thus resulting size
    print(df.dtypes)
    scope = kwargs.get('scope', 'all-downloaded')
    market = kwargs.get('market', 'asx')
    compression = kwargs.get('compression', 'snappy') 
    status = kwargs.get('status', 'FINAL')
    with io.BytesIO() as fp:
        # NB: if this fails it may be because you are using fastparquet 
        # which doesnt (yet) support BytesIO. Use eg. pyarrow
        df.to_parquet(fp, compression=compression, index=True)
        #table = Table.from_pandas(df, nthreads=1)
        #pq.write_table(table, fp, use_dictionary=True, compression=compression)
        fp.seek(0)
        byte_content = fp.read()
        size = len(byte_content)
        sha256_hash = hashlib.sha256(byte_content).hexdigest()
        db.market_quote_cache.update_one({
            'tag': tag, 'scope': scope}, {"$set": {
            'tag': tag, 'status': status,
            'last_updated': datetime.utcnow(),
            'field': field_name,
            'market': market,
            'scope': scope,
            'n_days': n_days,
            'n_stocks': n_stocks,
            'dataframe_format': 'parquet',
            'size_in_bytes': size,
            'sha256': sha256_hash,
            'dataframe': Binary(byte_content), # NB: always parquet format
        }}, upsert=True)
        print(f"{tag} == {size} bytes (sha256 hexdigest {sha256_hash})")

def load_all_fields(db, month: int, year: int, required_fields, **kwargs):
    assert len(required_fields) > 0
    db.market_quote_cache.create_index([('tag', pymongo.ASCENDING)], unique=True)
    uber_df = None
    impute_fields = kwargs.get('impute_fields', set())
    market = kwargs.get('market', 'asx')
    for field_name in required_fields:
        print("Constructing matrix: {} {}-{}".format(field_name, month, year))
        rows = load_field_dataframe(db, field_name, month, year)
        if len(rows) == 0:
            continue
        df = pd.DataFrame.from_records(rows)
        df = df.pivot(index='fetch_date', columns='asx_code', values='field_value')
        if field_name in impute_fields: # some fields will break the viewer if too many NaN
            before = df.isnull().sum().sum()
            print(f"Imputing {field_name} - before {before} missing values")
            df = df.fillna(method='pad').fillna(method='backfill')
            after = df.isnull().sum().sum()
            print(f"After imputation {field_name} - now {after} missing values")
        rows = []
        for asx_code, series in df.items():
            for fetch_date, field_value in series.items():
                rows.append({ 'asx_code': asx_code, 'fetch_date': fetch_date, 'field_value': field_value, 'field_name': field_name})
        #print(rows)
        
        if uber_df is None:
            uber_df = pd.DataFrame.from_records(rows)
        else:
            uber_df = uber_df.append(rows)
        if df.isnull().values.any():
            dates_with_missing = set([df.loc[the_date].isnull().any() for the_date in df.index])
            today = datetime.strftime(datetime.today(), "%Y-%m-%d")
            if today in dates_with_missing:
                print(f"WARNING: today's {field_name} matrix contains missing data! Continuing anyway.")
                print(df.loc[today].isnull().values.any())
                # FALLTHRU...
        tag = "{}-{:02d}-{}-{}".format(field_name, month, year, market)
        #save_dataframe(db, df, tag, field_name, status, market, scope, compression='gzip', n_days=len(df), n_stocks=len(df.columns))

    all_stocks = set(uber_df['asx_code'])
    print("Detected {} stocks during month ({} datapoints total)".format(len(all_stocks), len(uber_df)))
    uber_tag = "uber-{:02d}-{}-{}".format(month, year, market)
    all_fields = set(uber_df['field_name'])
    # TODO FIXME: should we drop missing values from the uber parquet? might save space... since we will get them back on load
    #print(all_fields)
    #print(required_fields)
    assert all_fields == set(required_fields)

    print("% missing values: ", ((uber_df.isnull() | uber_df.isna()).sum() * 100 / uber_df.index.size).round(2))
    n_stocks = uber_df['asx_code'].nunique()
    n_days = uber_df['fetch_date'].nunique()
    print(uber_df)
    save_dataframe(db, uber_df, uber_tag, 'uber', n_days=n_days, n_stocks=n_stocks, **kwargs)

if __name__ == "__main__":
    a = argparse.ArgumentParser(description="Construct and ingest db.asx_prices into parquet format month-by-month and persist to mongo")
    default_host = 'pi1'
    default_port = 27017
    default_db = 'asxtrade'
    default_access = 'read-write'
    default_user = 'rw'
    a.add_argument("--db",
                        help="Mongo host/ip to save to [{}]".format(default_host),
                        type=str, default=default_host)
    a.add_argument("--port",
                        help="TCP port to access mongo db [{}]".format(str(default_port)),
                        type=int, default=default_port)
    a.add_argument("--dbname",
                        help="Name on mongo DB to access [{}]".format(default_db),
                        type=str, default=default_db)
    dict_args = { 
        'help': "MongoDB RBAC username to use ({} access required) [{}]".format(default_access, default_user),
        'type': str,
        'required': True
    }
    if default_user is not None:
        dict_args.pop('required', None)
        dict_args.update({ 'default': default_user })
    a.add_argument("--dbuser", **dict_args)
    a.add_argument("--dbpassword", help="MongoDB password for user", type=str, required=True)
    a.add_argument("--month", help="Month of year 1..12", required=True, type=int)
    a.add_argument("--year", help="Year to load [2022]", default=2022, type=int)
    a.add_argument("--status", help="Status of matrix eg. INCOMPLETE or FINAL", required=True, type=str)
    a.add_argument("--past365", help="Make past 365 days dataframe for speedier asxtrade.py [False]", action="store_true")
    a.add_argument('--compression', help="Compression to use [snappy]", choices=('gzip', 'lz4', 'snappy'), default='snappy')
    a.add_argument('--market', help="Market [asx]", type=str, default='asx')
    args = a.parse_args()

    pwd = str(args.dbpassword)
    if pwd.startswith('$'):
        pwd = os.getenv(args.dbpassword[1:])
    mongo = pymongo.MongoClient(args.db, args.port, username=args.dbuser, password=pwd)
    db = mongo[args.dbname]

    required_fields = ['change_in_percent', 'last_price', 'change_price', \
                       'day_low_price', 'day_high_price', 'volume', 'eps', \
                       'pe', 'annual_dividend_yield', 'market_cap', 'number_of_shares']

    # NB: viewer can 404 if too much missing data. So we try to address this by forward-filling market_cap and
    # number_of_shares. But it is incorrect mathematically. Remove if you dont like it ;-) TODO FIXME
    impute_fields = set(['market_cap', 'number_of_shares'])    
    load_all_fields(db, args.month, args.year, required_fields, market=args.market, status=args.status, compression=args.compression, impute_fields=impute_fields)
    print("Run completed successfully.")
    exit(0)
