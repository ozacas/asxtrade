import argparse
import io
import pymongo
import pandas as pd
from utils import read_config

if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Fetch the specified dataframe by tag and save to disk"
    )
    args.add_argument(
        "--config",
        help="Configuration file to use [config.json]",
        type=str,
        default="config.json",
    )
    args.add_argument("--tag", help="Tag to report", type=str, required=True)
    args.add_argument(
        "--format", help="One of CSV, Excel or Parquet [CSV]", type=str, default="CSV"
    )
    args.add_argument("--out", help="File to output data to", type=str, required=True)
    a = args.parse_args()
    config, password = read_config(a.config)
    m = config.get("mongo")
    mongo = pymongo.MongoClient(
        m.get("host"), m.get("port"), username=m.get("user"), password=password
    )
    db = mongo[m.get("db")]

    result = db.market_data_cache.find_one({"tag": a.tag})
    if result is None:
        print(f"No dataset: {a.tag}")
        exit(1)
    with io.BytesIO(result["dataframe"]) as fp:
        df = pd.read_parquet(fp)
        if a.format.lower() == "excel":
            df.to_excel(a.out)
        elif a.format.lower() == "csv":
            df.to_csv(a.out)
        elif a.format.lower() == "parquet":
            df.to_parquet(a.out)
        else:
            raise NotImplementedError("Unsupported format: {}".format(a.format))

    exit(0)
