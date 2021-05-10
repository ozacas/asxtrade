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
        "SEC",  # dont have correct pd.read_csv settings for this dataset, load fails
        "STP",  # ditto
        "STS",  # ditto
        "ST1",  # discontinued dataset according to https://sdw.ecb.europa.eu/browseExplanation.do?node=9689721
    ]
)

# some datasets need a little help to be ingested correctly via pandas, extra options to pd.read_csv() are specified per flow_name
extra_csv_parms = {
    "SEC": {"dtype": {"SEC_ISSUING_SECTOR": "int"}},
    "STS": {"dtype": {"STS_SUFFIX": "int"}},
    "YC": {"dtype": {"COMPILATION": "unicode"}},
    "BSI": {
        "dtype": {
            "BS_COUNT_SECTOR": "str",
            "OBS_COM": "str",
            "UNIT_INDEX_BASE": "str",
            "COMPILATION": "str",
        }
    },
    "ICP": {
        "dtype": {
            "TIME_PERIOD": "str",
            "SOURCE_AGENCY": "str",
        }
    },
}
entrypoint = "https://sdw-wsrest.ecb.europa.eu/service/"  # Using protocol 'https'
resource = "data"  # The resource for data queries is always 'data'
parameters = {"startPeriod": "2020-01-01", "endPeriod": "2021-05-01"}


def save_code_list(
    db_collection, flow_name: str, cl_name: str, code_list, metadata_type: str
) -> int:
    n = 0
    print(f"save_code_list({flow_name}, {cl_name}, {metadata_type})")
    assert len(flow_name) > 0
    assert len(metadata_type) > 0
    assert len(cl_name) > 0
    try:
        df = sdmx.to_pandas(code_list).to_frame()
        for row in df.itertuples():
            d = {
                "flow_name": flow_name,
                "codelist_name": cl_name,
                "item_name": row.Index,
                "item_value": row._1,
                "metadata_type": metadata_type,
            }
            matcher = {
                "flow_name": flow_name,
                "codelist_name": cl_name,
                "item_name": row.Index,
                "metadata_type": metadata_type,
            }
            # print(matcher)
            result = db_collection.update_one(
                matcher,
                {"$set": d},
                upsert=True,
            )
            assert result is not None
            assert result.matched_count == 1 or result.upserted_id is not None
            n += 1
    except NotImplementedError:
        print(f"WARNING: unable to save_codelist: {cl_name} {flow_name}")
        # FALLTHRU...

    return n


def fetch_dataset(db_collection, flow_name: str, parameters):
    # 1. try CSV fetch, if that doesnt work then try PandasSDMX to get the dataframe
    data_response = requests.get(url, params=parameters, headers={"Accept": "text/csv"})
    assert data_response.status_code == 200
    with get_tempfile() as fp:
        fp.write(data_response.text.encode())
        fp.seek(0)
        kwargs = (
            {}
            if not flow_name in extra_csv_parms
            else dict(**extra_csv_parms[flow_name])
        )
        try:
            df = pd.read_csv(fp, **kwargs)
            save_dataframe(db_collection, {}, df, url, "ECB")
            return
        except pd.errors.EmptyDataError:  # no data is ignored as far as --fail-fast is concerned
            print(f"No CSV data to save.. now trying {flow_name} using pandasdmx")
            # FALLTHRU...

    # 2. try pandassdmx if CSV fetch fails
    ecb = sdmx.Request("ECB", backend="memory")
    data_msg = ecb.data(flow_name, params=parameters)
    df = sdmx.to_pandas(data_msg)
    assert isinstance(df, pd.DataFrame)
    save_dataframe(db_collection, {}, df, url, "ECB")


def update_flow(db, flow_name: str):
    assert db is not None
    assert len(flow_name) >= 2
    tag = f"https://sdw-wsrest.ecb.europa.eu/service/data/{flow_name}/"

    data_available = db.ecb_data_cache.find_one({"tag": tag, "scope": "ECB"})
    print(f"{flow_name} {data_available is None}")
    result = db.ecb_flow_index.update_one(
        {"flow_name": flow_name},
        {"$set": {"data_available": data_available is not None}},
    )
    assert isinstance(result, pymongo.results.UpdateResult)
    assert result.matched_count == 1


def save_flow(
    db_collection,
    header: sdmx.message.Header,
    flow_name: str,
    flow_descr: str,
    data_available: bool,
) -> None:
    d = {
        "prepared": header.prepared,
        "is_test_data": header.test,
        "source": str(header.source)
        if header.source is not None
        else None,  # use str() to ensure mongo string compatibility from SDMX type
        "sender": str(header.sender),
        "flow_name": flow_name,
        "flow_descr": flow_descr,
        "last_updated": now(),
        "data_available": data_available,
    }

    result = db_collection.update_one(
        {"flow_name": flow_name}, {"$set": d}, upsert=True
    )
    assert result is not None
    assert result.matched_count == 1 or result.upserted_id is not None


def save_metadata(db_collection, dsd, flow_name: str) -> int:
    assert db_collection is not None
    assert dsd is not None
    assert len(flow_name) > 0

    # 1. save dimension data
    for dim in dsd.dimensions.components:
        code_list = dim.local_representation.enumerated
        save_code_list(
            db_collection, flow_name, dim.id, code_list, metadata_type="dimension"
        )

    # 2. save attribute metadata
    for attr in dsd.attributes.components:
        result = attr.local_representation.enumerated
        # print(attr.id)
        if result is not None:
            save_code_list(
                db_collection, flow_name, attr.id, result, metadata_type="attribute"
            )

    # 3. save observation metadata
    for obs in dsd.measures.components:
        result = obs.local_representation.enumerated
        # print(obs.id)
        if result is not None:
            save_code_list(
                db_collection, flow_name, obs.id, result, metadata_type="measure"
            )

    return 0


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
        "--no-data-fetch",
        help="Do not fetch data ie. metadata only",
        action="store_true",
    )
    args.add_argument(
        "--always-update-metadata",
        help="Update metadata even if recently processed",
        action="store_true",
    )
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
    all_flows = ecb.dataflow()
    df = sdmx.to_pandas(all_flows.dataflow).to_frame()

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

            if not url in recent_tags or a.always_update_metadata:
                print(f"Updating metadata for {flow_name}")
                current_msg = ecb.dataflow(flow_name)
                current_flow = current_msg.dataflow[flow_name]
                dsd = current_flow.structure
                # print(current_msg.header)
                save_flow(
                    db.ecb_flow_index, current_msg.header, flow_name, flow_descr, False
                )
                save_metadata(db.ecb_metadata_index, dsd, flow_name)

            if not (a.no_data_fetch or url in recent_tags):
                fetch_dataset(db.ecb_data_cache, flow_name, parameters)

            if url in recent_tags:
                print(f"Skipping update to {url} since recently processed.")
            elif a.no_data_fetch:
                print(f"Skipping {url} due to --no-data-fetch")

            update_flow(db, flow_name)
            time.sleep(a.delay)
        except KeyboardInterrupt:
            exit(1)
        except Exception:
            print(f"WARNING: failed to download {flow_name}")
            update_flow(db, flow_name)
            traceback.print_exc()
            if a.fail_fast:
                raise

    exit(0)
