#!/usr/bin/python3.8
import argparse
from typing import Iterable
import requests
import pymongo
from bs4 import BeautifulSoup
from pandasdmx import Request
import pandas as pd
from datetime import timedelta
import traceback
import time
from requests.exceptions import HTTPError
from utils import save_dataframe, read_config, now

do_not_download = set(["ABS_BLDG_APPROVALS"])


def get_known_dataset_ids() -> Iterable[str]:
    resp = requests.get("http://stat.data.abs.gov.au/sdmx-json/")
    assert resp.status_code == 200
    soup = BeautifulSoup(resp.content, features="lxml")
    select = soup.find("select", {"name": "Datasets"})
    assert select is not None
    children = select.findChildren("option")
    results = []
    for child in children:
        dataset = child.get("value")
        assert len(dataset) > 0
        if dataset == "-":
            continue
        title = child.text
        results.append((dataset, title))
    print(f"Found {len(results)} datasets from ABS website")
    return results


if __name__ == "__main__":
    known_datasets = get_known_dataset_ids()
    ABS = Request("ABS")  # , log_level=logging.INFO)

    args = argparse.ArgumentParser(
        description="Load Australian Bureau of Statistics (ABS) datasets and create matching inverted index for abs app to use"
    )
    args.add_argument(
        "--config",
        help="Configuration file to use [config.json]",
        type=str,
        default="config.json",
    )
    args.add_argument(
        "--delay", help="Pause X seconds between requests [30]", type=int, default=30
    )
    args.add_argument(
        "--api-key",
        help="API Key to use to fetch ABS indicator API data",
        type=str,
        default=None,
    )
    a = args.parse_args()
    config, password = read_config(a.config)
    m = config.get("mongo")
    mongo = pymongo.MongoClient(
        m.get("host"), m.get("port"), username=m.get("user"), password=password
    )
    db = mongo[m.get("db")]

    month_ago = now() - timedelta(days=30)
    recent_data_sets = set(
        [
            r["field"]
            for r in db.abs_data_cache.find({"last_updated": {"$gte": month_ago}})
        ]
    )

    for dataset, title in filter(
        lambda t: t[0] not in do_not_download and t[0] not in recent_data_sets,
        known_datasets,
    ):
        print(f"Processing dataset: {dataset} {title}")
        try:
            data_response = ABS.data(
                resource_id=dataset, params={"startPeriod": "2010"}
            )
            df = data_response.write().unstack().reset_index()
            assert len(df) > 0 and isinstance(df, pd.DataFrame)
            tag = f"{dataset}-dataframe"
            metadata = {
                "dataset": dataset,
                "name": title,
                "tag": tag,
                "field": dataset,
                "scope": "abs",
                "last_updated": now(),
                "min_date": None,
                "max_date": None,
                "n_attributes": len(df.columns),
            }
            save_dataframe(db.abs_data_cache, metadata, df, tag, "abs")
            db.abs_inverted_index.update_one(
                {"dataset": dataset, "scope": "abs"}, {"$set": metadata}, upsert=True
            )
            time.sleep(a.delay)
        except HTTPError:
            print(f"WARNING: unable to fetch {dataset}")
            traceback.print_exc()

    exit(0)
