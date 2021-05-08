#!/usr/bin/python3
import pandasdmx as sdmx
import argparse
import requests
import pymongo
import time
import traceback
from datetime import timedelta
import pandas as pd
from utils import save_dataframe, read_config, now, get_tempfile

do_download = set()
do_not_download = set(
    [
        "IVF",  # uses large memory to ingest dataframe
        "RTD",  # research database
        "SAFE",  # survey dataset
        "SPF",  # survey dataset
    ]
)
extra_csv_parms = {
    "SEC": {"dtype": {"SEC_ISSUING_SECTOR": "int"}},
    "STS": {"dtype": {"STS_SUFFIX": "int"}},
    "YC": {"dtype": {"COMPILATION": "unicode"}},
}
entrypoint = "https://sdw-wsrest.ecb.europa.eu/service/"  # Using protocol 'https'
resource = "data"  # The resource for data queries is always 'data'
parameters = {"startPeriod": "2020-01-01", "endPeriod": "2021-05-01"}

if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Update financial performance metrics for ASX stocks using yfinance"
    )
    args.add_argument(
        "--config",
        help="Configuration file to use [config.json]",
        type=str,
        default="config.json",
    )
    args.add_argument("--fail-fast", help="Stop on first error", action="store_true")
    args.add_argument("--all", help="Download all ECB datasets", action="store_true")
    args.add_argument(
        "--delay", help="Delay between datasets in seconds [30]", type=int, default=30
    )
    args.add_argument(
        "--dataset",
        help="Download only the specified flowRef [None]",
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

    ecb = sdmx.Request("ECB")
    resp = ecb.dataflow()
    df = sdmx.to_pandas(resp.dataflow).to_frame()
    print(f"Found {len(df)} ECB datasets to process")

    if a.dataset:
        for s in a.dataset.split(","):
            do_download.add(s)
        recent_tags = set()  # forcibly re-download all stated datasets
    else:
        month_ago = now() - timedelta(days=30)
        recent_tags = set(
            db.ecb_data_cache.distinct("tag", {"last_updated": {"$gte": month_ago}})
        )

    for flow_name, flow_descr in df.itertuples():
        try:
            if (flow_name in do_download or a.all) and not flow_name in do_not_download:
                print(f"Processing {flow_name}: {flow_descr}")
            else:
                continue
            key = ""
            url = entrypoint + resource + "/" + flow_name + "/" + key
            if url in recent_tags:
                print(f"Skipping update to {url} since recently processed.")
                continue
            data_response = requests.get(
                url, params=parameters, headers={"Accept": "text/csv"}
            )
            assert data_response.status_code == 200
            with get_tempfile() as fp:
                fp.write(data_response.text.encode())
                fp.seek(0)
                kwargs = (
                    {}
                    if not flow_name in extra_csv_parms
                    else dict(**extra_csv_parms[flow_name])
                )
                df = pd.read_csv(fp, **kwargs)
                # print(df)
                save_dataframe(db.ecb_data_cache, {}, df, url, "ECB")
            time.sleep(a.delay)
        except KeyboardInterrupt:
            exit(1)
        except Exception:
            print(f"WARNING: failed to download {flow_name}")
            traceback.print_exc()
            if a.fail_fast:
                raise

    exit(0)
