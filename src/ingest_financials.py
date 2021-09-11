#!/usr/bin/python3
"""
Responsible for ingesting data related to the business performance over time. Data is placed into the asx_company_financial_metric
collection, ready for the core viewer app to use. Stocks whose financial details have been retrieved in the past month are skipped.
"""
import pymongo
import argparse
import yfinance as yf
import time
from utils import read_config
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bson.objectid import ObjectId


def melt_dataframes(dfs: tuple) -> pd.DataFrame:
    result = None
    for df in filter(lambda df: df is not None and len(df) > 0, dfs):
        df["metric"] = df.index
        melted = pd.melt(df, id_vars=("metric"), var_name="date")
        melted = melted.dropna(axis=0, how="any")
        if len(melted) == 0:
            continue
        # print(melted)
        # print(melted.shape)
        if result is None:
            result = melted
        else:
            result = result.append(melted)
    if result is not None and "date" in result.columns:
        # print(result)
        result["date"] = pd.to_datetime(
            result["date"], infer_datetime_format=True
        )  # format="%Y-%m-%d")
    # print(result)
    return result


def desired_stocks():
    available_stocks = set(db.asx_company_details.distinct("asx_code"))
    gen_time = datetime.today() - timedelta(days=30)
    month_ago = ObjectId.from_datetime(gen_time)
    recently_updated_stocks = set(
        [
            rec["asx_code"]
            for rec in db.asx_company_financial_metrics.find(
                {"_id": {"$gte": month_ago}}
            )
        ]
    )

    return available_stocks.difference(recently_updated_stocks)


def update_all_metrics(df: pd.DataFrame, asx_code: str) -> int:
    """
    Add (or update) all financial metrics (ie. rows) for the specified asx_code in the specified dataframe
    :rtype: the number of records updated/created is returned
    """
    print(f"Updating {len(df)} financial metrics for {asx_code}")
    n = 0
    for t in df.itertuples():
        d = {
            "metric": t.metric,
            "date": t.date,
            "value": t.value,
            "asx_code": t.asx_code,
        }
        assert t.asx_code == asx_code
        result = db.asx_company_financial_metrics.update_one(
            {"asx_code": asx_code, "date": t.date, "metric": t.metric},
            {"$set": d},
            upsert=True,
        )
        assert result is not None
        assert isinstance(result, pymongo.results.UpdateResult)
        assert result.matched_count == 1 or result.upserted_id is not None
        n += 1
    return n


def fetch_metrics(asx_code: str) -> pd.DataFrame:
    """
    Using the excellent yfinance, we fetch all possible metrics of business performance for the specified stock code.
    Returns a dataframe (possibly empty or none) representing each metric and its datapoints as separate rows
    """
    assert len(asx_code) >= 3
    ticker = yf.Ticker(asx_code + ".AX")
    cashflow_df = ticker.cashflow
    financial_df = ticker.financials
    earnings_df = ticker.earnings
    if set(earnings_df.columns) == set(["Earnings", "Revenue"]):
        earnings_df.index = earnings_df.index.map(
            str
        )  # convert years to str (maybe int)
        earnings_df = earnings_df.transpose()

    # print(earnings_df)
    balance_sheet_df = ticker.balance_sheet
    melted_df = melt_dataframes(
        (cashflow_df, financial_df, earnings_df, balance_sheet_df)
    )
    return melted_df


def make_asx_prices_dict(new_quote: tuple, asx_code: str) -> dict:
    #print(new_quote)

    d = {
        "asx_code": asx_code,
        "fetch_date": new_quote.Index,
        "volume": new_quote.Volume,
        "last_price": new_quote.Close,
        "day_low_price": new_quote.Low,
        "day_high_price": new_quote.High,
        "open_price": new_quote.Open,
        "error_code": "",
        "error_descr": "",
        # we dont set nan fields so that existing values (if any) are used ie. merge with existing data
        # "annual_dividend_yield": np.nan,  # no available data from yf.Ticker.history() although may be available elsewhere, but for now set to missing
        # "annual_daily_volume": np.nan,
        # "bid_price": np.nan,
        "change_price": new_quote.change_price,
        "change_in_percent": new_quote.change_in_percent,
    }
    return d


def fill_stock_quote_gaps(db, stock_to_fetch: str, force=False) -> int:
    assert db is not None
    assert len(stock_to_fetch) >= 3
    ticker = yf.Ticker(stock_to_fetch + ".AX")
    df = ticker.history(period="max")
    df.index = [d.strftime("%Y-%m-%d") for d in df.index]
    # print(df)
    available_dates = set(df.index)
    available_quotes = list(db.asx_prices.find({"asx_code": stock_to_fetch}))
    quoted_dates = set(
        [q["fetch_date"] for q in available_quotes if not np.isnan(q["last_price"])]
    )
    assert set(df.columns) == set(
        ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
    )
    dates_to_fill = (
        available_dates.difference(quoted_dates) if not force else available_dates
    )
    print(
        "Got {} existing daily quotes for {}, found {} yfinance daily quotes, gap filling for {} dates (force={})".format(
            len(available_quotes), stock_to_fetch, len(df), len(dates_to_fill), force
        )
    )
    if len(dates_to_fill) < 1:
        return 0

    df["change_price"] = df["Close"].diff()
    df["change_in_percent"] = df["Close"].pct_change() * 100.0
    gap_quotes_df = df.filter(dates_to_fill, axis=0)
    # print(df)
    n = 0
    for new_quote in gap_quotes_df.itertuples():
        d = make_asx_prices_dict(new_quote, stock_to_fetch)
        result = db.asx_prices.update_one(
            {"fetch_date": d["fetch_date"], "asx_code": d["asx_code"]},
            {"$set": d},
            upsert=True,
        )
        assert result is not None

        # assert result.modified_count == 1 or result.upserted_id is not None
        n += 1
    assert n == len(gap_quotes_df)
    return n


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
    args.add_argument(
        "--fill-gaps",
        help="Fill dates with no existing quotes for each stock (use --debug for a particular stock)",
        action="store_true",
    )
    args.add_argument("--fail-fast", help="Stop on first error", action="store_true")
    args.add_argument(
        "--delay", help="Delay between stocks in seconds [30]", type=int, default=30
    )
    args.add_argument("--force", help="Overwrite existing data (if any)", action="store_true")
    args.add_argument(
        "--debug",
        help="Try to fetch specified stock (for debugging)",
        type=str,
        required=False,
        default=None,
    )
    a = args.parse_args()
    config, password = read_config(a.config)
    m = config.get("mongo")
    mongo = pymongo.MongoClient(
        m.get("host"), m.get("port"), username=m.get("user"), password=password
    )
    db = mongo[m.get("db")]

    stock_codes = desired_stocks() if not a.debug else set([a.debug])
    print(f"Updating financial metrics for {len(stock_codes)} stocks")
    for asx_code in sorted(stock_codes):
        print(f"Processing stock {asx_code}")
        try:
            melted_df = fetch_metrics(asx_code)
            if melted_df is None or len(melted_df) < 1:
                raise ValueError(f"No data available for {asx_code}... skipping")
            melted_df["asx_code"] = asx_code
            ret = update_all_metrics(melted_df, asx_code)
            assert ret == len(melted_df)
            if a.fill_gaps:
                fill_stock_quote_gaps(db, asx_code, force=a.force)
                # FALLTHRU...
            time.sleep(a.delay)
        except Exception as e:
            print(f"WARNING: unable to download financials for {asx_code}")
            print(str(e))
            if a.fail_fast:
                raise e

    exit(0)
