#!/usr/bin/env python3
"""
Responsible for loading historical ASX prices and converting them into django model equivalents.
Not all data can be provided: fields which are missing are set to None. Historical prices are loaded
from google finance API using pandas_data_reader. This script is carefully written to not alter 
data already existing in the specified database.
"""
import argparse
import time
import json
import pandas as pd
import pymongo
import re
import wbdata as wb
from typing import Iterable
from bson.binary import Binary
from datetime import datetime, timedelta
from utils import read_config, save_dataframe, now


# id = ObjectIdField(primary_key=True)
# wb_id = model.TextField()  # world bank id, not to be confused with django id/pk
# name = model.TextField()
# last_updated = model.DateTimeField(auto_now_add=True)
# unit = model.TextField()
# source = JSONField() 
# source_note = model.TextField()
# topics = ArrayField(WorldBankTopics)
# source_organisation = model.TextField()

# # errors associated with fetching data
# last_successful_data = model.DateTimeField(null=True) # null denotes no data fetched for this indicator
# last_error_when = model.DateTimeField(null=True) # null denotes no error encountered via this indicator
# last_error_msg = model.TextField(null=True)
# last_error_type = model.TextField()

    # @property
    # def tag(self):
    #     return f"{self.wb_id}-dataframe"

def as_tag(i: dict, freq: str) -> str:
    key = i['wb_id']
    assert key is not None and len(key) > 0
    if freq == 'Y':
        printable_freq = 'yearly'
    elif freq == 'M':
        printable_freq = 'monthly'
    elif freq == 'Q':
        printable_freq = 'quarterly'
    else:
        print(freq)
        assert False
    return f"{key}-{printable_freq}-dataframe"

def fix_dataframe(i: dict, df: pd.DataFrame, countries, tag: str) -> tuple:
    df = df.reset_index(drop=False)
    attributes = set(df.columns)
    attribute_metadata = { 'n_attributes': len(attributes) }
    for key in ['date', 'country']:
        attribute_metadata['has_'+key] = key in attributes
    # remove rows with bad dates which we cant plot right now... TODO FIME:
    bad_dates = []
    for unique_date in df['date'].unique():
        if not re.match(r'^\d{4}$', unique_date):
            bad_dates.append(unique_date)
    if len(bad_dates) > 0:
        print("WARNING: removing bad date entries from dataframe: {}".format(bad_dates)) # TODO FIXME... something smarter than this
        df = df[~df['date'].isin(bad_dates)]
    #print(df['date'].unique())
    if 'date' in attributes:
        mean_date_len = df['date'].apply(lambda v: len(str(v))).mean()
        n_dates = df['date'].nunique()
        attribute_metadata['mean_date_len_in_chars'] = mean_date_len
        df['date'] = pd.to_datetime(df['date'])
        #print(mean_date_len)
        print(f"{n_dates} unique dates in {tag}")
        attribute_metadata['n_unique_dates'] = n_dates
    else:
        attribute_metadata['n_unique_dates'] = 0
        print("WARNING: no date in dataframe: {}".format(attributes))

    attribute_metadata['n_row'] = len(df)    # includes NA
    df = df.dropna(subset=['metric'], how='any')
    attribute_metadata['n_row_not_na'] = len(df)
    if 'country' in attributes:
        n_countries = df['country'].nunique()
        print(f"Found {n_countries} countries in {tag}")
        attribute_metadata['n_countries'] = n_countries
    else:
        print("WARNING: no country in dataframe: {}".format(attributes))
    return df, attribute_metadata

def get_topics(indicator: dict):
    topics_for_indicator = indicator.get('topics', None)
    if topics_for_indicator is None:
        return []

    if not isinstance(topics_for_indicator, list):
        topics_for_indicator = json.loads(topics_for_indicator)
    return topics_for_indicator

def has_topics(indicator: dict):
    l = get_topics(indicator)
    return len(l) > 0

def save_inverted_index(db, metadata: dict, dataframe: pd.DataFrame, indicator: dict, tag: str, countries: dict) -> int:
    df_countries = set(dataframe['country'].unique())
    n_topics = n_countries = 0
    n_updated = 0
   
    for t in get_topics(indicator):
        n_topics += 1
        for df_c in df_countries:
            n_countries += 1
            try:
                d = dict(**metadata)
                topic_id = int(t['id'])
                c = countries[df_c]
            except KeyError:
                print(f"Not indexing unknown country/region or topic: {df_c} {t}")
                continue
            country_code = c.get('country_code', None)
            d['country'] = country_code
            d['tag'] = tag
            d['topic_id'] = topic_id
            if len(t['id']) <= 0:
                raise ValueError(f"No topic for {indicator}")
            #print(t)
            d['topic_name'] = t.get('value', t.get('topic', None))
            d['last_updated'] = now()
            #print(d)
            #print(indicator)
            assert isinstance(topic_id, int)
            assert isinstance(country_code, str) and len(country_code) <= 3 and len(country_code) > 0
            assert len(tag) > 0 and isinstance(tag, str)
            assert topic_id > 0
            assert isinstance(d['topic_name'], str)
            assert len(d['topic_name']) > 0
            d['indicator_id'] = i['wb_id']
            d['indicator_name'] = i['name']
            db.world_bank_inverted_index.update_one({ 'scope': 'worldbank', 'tag': tag, 'country': country_code, 'topic_id': topic_id }, 
                                                    { "$set": d }, upsert=True)
            n_updated += 1

    if n_topics == 0:
        raise ValueError(f"No topics for {tag} -- ignoring from dataset index")
    if n_countries == 0:
        raise ValueError(f"No countries for {tag} -- ignoring from dataset index")
    return n_updated
    

def purge_all_worldbank_data(db) -> None:
    db.world_bank_countries.drop()
    db.world_bank_indicators.drop()
    db.world_bank_topics.drop()
    db.world_bank_inverted_index.drop()
    db.market_data_cache.remove({'scope': 'worldbank'})
    print("All existing worldbank data removed from DB")


def purge_obsolete_data(db, current_indicators: Iterable[str], known_indicators: Iterable[str]) -> None:
    current_tags = set(current_indicators.keys())
    obsolete_tags = set(known_indicators.keys()).difference(current_tags)
    print("{} datasets are no longer served by worldbank website, removing due to user request".format(len(obsolete_tags)))

    if len(obsolete_tags) > 0:
        #db.world_bank_indicators.remove({ 'wb_id': {'$in': obsolete_tags}, 'scope': 'worldbank'})
        print("Removed obsolete indicators")
    obsolete_dataframes = set(db.market_data_cache.distinct('field')).difference(current_tags)
    if len(obsolete_dataframes) > 0:
        #db.market_data_cache.remove({'field': {'$in': obsolete_dataframes}, 'scope': 'worldbank'})
        print("Removed obsolete dataframes")

def update_indicator(db, indicator: dict) -> None:
    assert db is not None
    assert isinstance(indicator, dict) and len(indicator.keys()) > 0
    indicator.pop('_id', None)
    result = db.world_bank_indicators.update_one({ "wb_id": i['wb_id'], "scope": "worldbank" }, { "$set": indicator }, upsert=True)
    assert isinstance(result, pymongo.results.UpdateResult)
    assert result.upserted_id is not None or result.modified_count == 1
    #print(result)

def all_countries(db):
    countries = {}
    for row in wb.get_country():
        assert isinstance(row, dict)
        d = { 'country_code': row['id'], 'name': row['name'], 'last_updated': now()} # none of the other state in row is saved in the model for now eg. incomeLevel/region
        db.world_bank_countries.update_one({ 'country_code': row['id'] }, { "$set": d }, upsert=True)
        countries[d['name']] = d
    return countries

def all_topics(db):
    topics = []
    for row in wb.get_topic():
        assert isinstance(row, dict)
        #print(row)
        d = { 'id': int(row['id']), 'last_updated': now(), 'topic': row['value'], 'source_note': row['sourceNote']}
        db.world_bank_topics.update_one({ 'id': d['id']}, {"$set": d }, upsert=True)
        topics.append(d)

    return topics

def all_indicators(db, update_local_db=False):
    metrics = {}
    
    for row in wb.get_indicator():
        assert isinstance(row, dict)
        assert isinstance(row['sourceNote'], str) 
        assert isinstance(row['topics'], list)
        assert isinstance(row['sourceOrganization'], str)
        assert isinstance(row['name'], str)
        #print(row)
        d = {
            'last_updated': now(),
            'wb_id': row['id'],
            'source_note': row['sourceNote'],
            'name': row['name'],
            'unit': row['unit'],
            'source_organisation': row['sourceOrganization'],
            'topics': row['topics']
        }
        if update_local_db:
            db.world_bank_indicators.update_one({ 'wb_id': d['wb_id'] }, { "$set": d}, upsert=True)
        metrics[d['wb_id']] = d

    print("Found {} datasets from wbdata".format(len(metrics.keys())))
    return metrics

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Load workbank API datasets and create matching inverted index for viewer app to use")
    args.add_argument('--config', help="Configuration file to use [config.json]", type=str, default="config.json")
    args.add_argument('--delay', help="Delay between datasets (seconds) [30]", type=int, default=30)
    args.add_argument('--clean', help="Remove existing data downloads and all indexes [no]", action='store_true')
    args.add_argument('--rm-obsolete', help="Remove indicators from database which are not known anymore [False]", action='store_true')
    args.add_argument('--freq', help="One of: Y==yearly, M==monthly, Q=quarterly. Not all datasets support all options", default='Y')
    args.add_argument('--all', help="Re-process all datasets inc. those processed in last month", action='store_true')
    args.add_argument('--prefix', help="Limit ingestion to indicators prefixed with specified string (case-sensitive) []", type=str, default=None)
    args.add_argument('--fail-fast', help="Stop on first error", action='store_true')
    a = args.parse_args()
    config, password = read_config(a.config)
    m = config.get('mongo')
    mongo = pymongo.MongoClient(m.get('host'), m.get('port'), username=m.get('user'), password=password)
    db = mongo[m.get('db')]

    countries = all_countries(db)
    topics = all_topics(db)
    known_indicators = { i['wb_id']: i for i in db.world_bank_indicators.find({}) }
    current_indicators = all_indicators(db, update_local_db=a.all)
    indicators = current_indicators if a.all else known_indicators
    if a.rm_obsolete:
        purge_obsolete_data(db, current_indicators, known_indicators)
    if a.clean:
        purge_all_worldbank_data(db)

    print("Found {} countries, {} topics and {} indicators.".format(len(countries), len(topics), len(indicators)))

    month_ago = datetime.utcnow() - timedelta(days=30)
    recent_tags = set([i['tag'] for i in db.world_bank_inverted_index.find({}) if i['last_updated'] > month_ago])
    recent_error_tags = [as_tag(i, 'Y') for i in db.world_bank_indicators.find({'last_error_when': {'$gt': month_ago }})]
    if len(recent_error_tags) > 0:
        print("Skipping {} datasets which failed to be downloaded in past month".format(len(recent_error_tags)))
        recent_tags = recent_tags.union(recent_error_tags)

    if not a.all:
        print("Ignoring {} datasets which have been downloaded and indexed in past month".format(len(recent_tags)))
   
    # https://github.com/OliverSherouse/wbdata/issues/23 suggested fix as caching by default yields poor performance
    def inner():
        pass
    wb.fetcher.CACHE.sync = inner
    # ensure full text search is possible at a future date
    db.world_bank_indicators.create_index([('name', 'text'), ('source_note', 'text')]) 
    n_downloaded = 0
    print("Processing {} datasets...".format(len(indicators.keys())))
    # TODO FIXME: add transaction support using pymongo
    if a.prefix is not None:
        filter_func = lambda t: t[0].startswith(a.prefix)
    else:
        filter_func = lambda t: True  # no filtering
    for wb_id, i in filter(filter_func, indicators.items()):
        if not has_topics(i):
            print(f"WARNING: skipping dataset {wb_id} as it has no topics")
            continue

        tag = as_tag(i, a.freq)

        if tag in recent_tags and not a.all:
            print(f"Skipping recently updated dataset: {wb_id}")
            continue
        try:
            print("Fetching... {} {}".format(wb_id, i['name']))
            df = wb.get_dataframe({wb_id: "metric"}, freq=a.freq)
            if df is None:
                raise ValueError('No data')
            df, metadata = fix_dataframe(i, df, countries, tag)
            #print(metadata)
            if df is None or len(df) == 0:
                print(f"WARNING: no data associated with {wb_id}")
                continue

            save_dataframe(db.market_data_cache, i, df, tag, 'worldbank')
            
            n = save_inverted_index(db, metadata, df, i, tag, countries)
            print(f"Updated {n} records for {tag}")
            as_at = now()
            i.update({
                'last_successful_data': as_at,
                'last_updated': as_at,
            })
            update_indicator(db, i)
            time.sleep(a.delay)
            n_downloaded += 1
        except (RuntimeError, ValueError, TypeError) as e:
            print(f"ERROR: when processing {i}: {e}")
            i.update({
                'last_error_when': now(),
                'last_error_msg': str(e),
                'last_error_type': str(type(e))
            })
            update_indicator(db, i)
            if a.fail_fast:
                raise e
    print(f"Updated {n_downloaded} datasets. Run completed.")
    exit(0)
