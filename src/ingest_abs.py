#!/usr/bin/python3.8
import argparse
from typing import Iterable
import requests
import pymongo
from bs4 import BeautifulSoup
from pandasdmx import Request
import pandas as pd
from utils import save_dataframe, read_config, now

def get_known_dataset_ids() -> Iterable[str]:
    resp = requests.get('http://stat.data.abs.gov.au/sdmx-json/')
    assert resp.status_code == 200
    soup = BeautifulSoup(resp.content, features="lxml")
    select = soup.find('select', { 'name': 'Datasets'})
    assert select is not None
    children = select.findChildren('option')
    results = []
    for child in children:
        dataset = child.get('value')
        assert len(dataset) > 0
        if dataset == '-':
            continue
        title = child.text
        results.append((dataset, title))
    print(f"Found {len(results)} datasets from ABS website")
    return results

if __name__ == "__main__":
    known_datasets = get_known_dataset_ids()
    ABS = Request('ABS')  # , log_level=logging.INFO)

    args = argparse.ArgumentParser(description="Load Australian Bureau of Statistics (ABS) datasets and create matching inverted index for abs app to use")
    args.add_argument('--config', help="Configuration file to use [config.json]", type=str, default="config.json")
    a = args.parse_args()
    config, password = read_config(a.config)
    m = config.get('mongo')
    mongo = pymongo.MongoClient(m.get('host'), m.get('port'), username=m.get('user'), password=password)
    db = mongo[m.get('db')]

    for dataset, title in known_datasets:
        print(f"Processing dataset: {dataset} {title}")
        data_response = ABS.data(resource_id=dataset, params={'startPeriod': '1990'})
        df = data_response.write().unstack().reset_index()
        assert len(df) > 0 and isinstance(df, pd.DataFrame)
        tag = f"{dataset}-dataframe"
        metadata = {
            'dataset': dataset,
            'name': title,
            'tag': tag,
            'scope': 'abs',
            'last_updated': now(),
            'min_date': None,
            'max_date': None,
            'n_attributes': len(df.columns),
        }
        save_dataframe(db.abs_data_cache, metadata, df, tag, 'abs')
        db.abs_inverted_index.update_one({ 'dataset': dataset, 'scope': 'abs' }, { '$set': metadata }, upsert=True)
        exit(0)
