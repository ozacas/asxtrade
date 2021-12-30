#!/usr/bin/python3.8
import argparse
import dateutil
import json
import time
import requests
from random import randint
from datetime import datetime, timedelta, timezone
import pymongo
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from utils import *

import pandas as pd
import os
import re

retry_strategy = Retry(
    total=10,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS"],
    backoff_factor=5,
)
retry_adapter = HTTPAdapter(max_retries=retry_strategy)


def update_companies(db, config, ensure_indexes=True):
    resp = requests.get(config.get("asx_companies"))
    if ensure_indexes:
        db.companies.create_index([("asx_code", pymongo.ASCENDING)], unique=True)

    fname = "{}/companies.{}.csv".format(
        config.get("data_root"), datetime.now().strftime("%Y-%m-%d")
    )
    df = None
    n = 0
    for line in resp.text.splitlines():
        if any(
            [
                line.startswith("ASX listed companies"),
                len(line.strip()) < 1,
                line.startswith("Company name"),
            ]
        ):
            continue
        fields = line.split(
            '","'
        )  # must use 3 chars here as just splitting on comma will break commas used in sector names (for example)
        assert len(fields) == 3
        d = {
            "asx_code": fields[1].strip('"'),
            "name": fields[0].strip('"'),
            "sector": fields[2].strip('"'),
            "last_updated": datetime.utcnow(),
        }
        assert len(d.get("asx_code")) >= 3
        assert len(d.get("name")) > 0
        if df is None:
            df = pd.DataFrame(columns=d.keys())
        row = pd.Series(d, name=d.get("asx_code"))
        df = df.append(row)
        db.companies.find_one_and_update(
            {"asx_code": d["asx_code"]}, {"$set": d}, upsert=True
        )
        n += 1
    df.to_csv(fname, sep="\t")
    print(
        "Saved {} companies to {} for validation by great_expectations.".format(
            n, fname
        )
    )


def update_isin(db, config, ensure_indexes=True):
    resp = requests.get(config.get("asx_isin"))
    if ensure_indexes:
        db.asx_isin.create_index(
            [("asx_code", pymongo.ASCENDING), ("asx_isin_code", pymongo.ASCENDING)],
            unique=True,
        )

    df = None
    with get_tempfile() as content:
        content.write(resp.content)
        content.seek(0)
        df = pd.read_excel(content.name)
        print(df.describe())

    fname = "{}/asx_isin/isin.{}.csv".format(
        config.get("data_root"), datetime.now().strftime("%Y-%m-%d")
    )
    all_records = []
    n = 0
    for row in df[4:].itertuples():
        # NB: first four rows are rubbish so we skip them during save...
        row_index, asx_code, company_name, security_name, isin_code = row
        asx_code = str(
            asx_code
        )  # some ASX codes are all integers which we dont want treated as int
        assert len(asx_code) >= 3
        assert len(company_name) > 0
        assert len(security_name) > 0
        d = {
            "asx_code": asx_code,
            "company_name": company_name,
            "security_name": security_name,
            "asx_isin_code": isin_code,
            "last_updated": datetime.utcnow(),
        }
        all_records.append(d)
        db.asx_isin.find_one_and_update(
            {"asx_isin_code": isin_code, "asx_code": asx_code}, {"$set": d}, upsert=True
        )
        n += 1
    out_df = pd.DataFrame.from_records(all_records)
    out_df.to_csv(fname, sep="\t")
    print("Saved {} securities to {} for validation.".format(n, fname))


def get_fetcher():
    fetcher = requests.Session()
    fetcher.mount("https://", retry_adapter)
    fetcher.mount("http://", retry_adapter)
    return fetcher


def validate_prices(dataframe):
    assert dataframe is not None
    import great_expectations as ge

    if isinstance(dataframe, str):  # TSV filename?
        dataframe = pd.read_csv(dataframe, sep="\t")
    try:
        context = ge.DataContext()
        print(context.get_available_data_asset_names())
    except:
        pass
    # context.


def update_prices(db, available_stocks, config, fetch_date, ensure_indexes=True):
    assert isinstance(config, dict)
    # assert len(available_stocks) > 10 # dont do this anymore, since we might have to refetch a few failed stocks

    if ensure_indexes:
        db.asx_prices.create_index(
            [("asx_code", pymongo.ASCENDING), ("fetch_date", pymongo.ASCENDING)],
            unique=True,
        )

    fetcher = get_fetcher()
    df = None
    print("Updating stock prices for {}".format(fetch_date))
    for asx_code in available_stocks:
        url = "{}{}{}".format(
            config.get("asx_prices"),
            "" if config.get("asx_prices").endswith("/") else "/",
            asx_code,
        )
        print("Fetching {} prices from {}".format(asx_code, url))
        already_fetched_doc = db.asx_prices.find_one(
            {"asx_code": asx_code, "fetch_date": fetch_date}
        )
        if already_fetched_doc is not None:
            print("Already got data for ASX {}".format(asx_code))
            continue
        try:
            resp = fetcher.get(url, timeout=(30, 30))
            if resp.status_code != 200:
                if (
                    resp.status_code == 404
                ):  # not found? ok, add it to blacklist... but we will check it again in future in case API broken...
                    db.asx_blacklist.find_one_and_update(
                        {"asx_code": asx_code},
                        {
                            "$set": {
                                "asx_code": asx_code,
                                "reason": "404 for {}".format(url),
                                "valid_until": datetime.utcnow()
                                + timedelta(days=randint(30, 60)),
                            }
                        },
                        upsert=True,
                    )
                raise ValueError(
                    "Got non-OK status for {}: {}".format(url, resp.status_code)
                )
            d = json.loads(resp.content.decode())
            d.update({"fetch_date": fetch_date})
            for key in ["last_trade_date", "year_high_date", "year_low_date"]:
                if key in d:
                    d[key] = dateutil.parser.parse(
                        d[key]
                    )  # NB: gonna be slow since the auto-format magic has to do it thing... but safer for common formats
            # assert len(d.keys()) > 10
            fix_percentage(d, "change_in_percent")
            fix_percentage(d, "previous_day_percentage_change")
            d["descr_full"] = d.pop(
                "desc_full", None
            )  # rename to correspond to django model
            # print(d)
            if df is None:
                df = pd.DataFrame(columns=d.keys())
            row = pd.Series(d, name=asx_code)
            df = df.append(row)
            assert (
                "descr_full" in d
            )  # renamed in django app so we must do the same here...
            db.asx_prices.find_one_and_update(
                {"asx_code": asx_code, "fetch_date": fetch_date},
                {"$set": d},
                upsert=True,
            )
        except AssertionError:
            raise
        except Exception as e:
            print("WARNING: unable to fetch data for {} -- ignored.".format(asx_code))
            print(str(e))
        time.sleep(5)  # be nice to the API endpoint
    fname = "{}/asx_prices/prices.{}.tsv".format(config.get("data_root"), fetch_date)
    if df is not None:
        df.to_csv(fname, sep="\t")
        validate_prices(df)
        print("Saved {} stock codes with prices to {}".format(len(df), fname))


def available_stocks(db, config):
    assert config is not None
    # only variants which include ORDINARY FULLY PAID/STAPLED SECURITIES eg. SYD
    stocks = [
        re.compile(r".*ORDINARY.*"),
        re.compile(r"^EXCHANGE\s+TRADED\s+FUND.*$"),
        re.compile(r"\bETF\b"),
        re.compile(r"^ETFS\b"),
        re.compile(r"\bETF\s+UNITS\b"),
        re.compile(r"\sETF$"),
        re.compile(r"\sFUND$"),
    ]
    ret = set()
    for security_type in stocks:
        tmp = set(
            [
                r.get("asx_code")
                for r in db.asx_isin.find({"security_name": security_type})
                if db.asx_blacklist.find_one(
                    {
                        "asx_code": r.get("asx_code"),
                        "valid_until": {"$gt": datetime.utcnow()},
                    }
                )
                is None
            ]
        )
        ret = ret.union(tmp)
    exclude_stocks_without_details = config.get("exclude_stocks_without_details", False)
    exclude_stocks_with_zero_volume = config.get("exclude_zero_volume_stocks", False)
    if (
        exclude_stocks_without_details
    ):  # will exclude if True nearly 1000 securities, so use with care
        details_stocks = set(db.asx_company_details.distinct("asx_code"))
        print(
            "Eliminating securities without company details: found {} to check".format(
                len(ret)
            )
        )
        ret = ret.intersection(details_stocks)
        # FALLTHRU...
    print("Found {} suitable stocks on ASX...".format(len(ret)))
    return sorted(ret)


def update_blacklist(db):
    assert db is not None
    bad_codes = {}
    for bad in db.asx_prices.find({"error_code": "id-or-code-invalid"}):
        asx_code = bad.get("asx_code")
        if not asx_code in bad_codes:
            bad_codes[asx_code] = set()
        bad_codes[asx_code].add(bad["fetch_date"])
    for_blacklisting = [code for code, dates in bad_codes.items() if len(dates) > 20]
    print(
        "Identified {} stocks for blacklisting with over 20 failed fetches".format(
            len(for_blacklisting)
        )
    )
    for code in for_blacklisting:
        db.asx_blacklist.find_one_and_update(
            {"asx_code": code},
            {"$set": {"asx_code": code, "reason": "asxtrade.py says no"}},
            upsert=True,
        )
    print("Blacklist updated.")


def update_company_details(db, available_stocks, config, ensure_indexes=False):
    assert len(available_stocks) > 10
    assert db is not None
    assert isinstance(config, dict)
    fetcher = get_fetcher()

    if ensure_indexes:
        db.asx_company_details.create_index(
            [
                (
                    "asx_code",
                    pymongo.ASCENDING,
                ),
            ],
            unique=True,
        )

    utc_now = datetime.now(timezone.utc)
    for asx_code in available_stocks:
        url = config.get("asx_company_details")
        url = url.replace("%s", asx_code)
        print(url)
        # rec should only have one record since we remove&insert per asx_code below...
        rec = [
            r
            for r in db.asx_company_details.find({"asx_code": asx_code})
            .sort([("_id", pymongo.ASCENDING)])
            .limit(50)
        ]
        if len(rec) > 0:
            # print(rec)
            dt = rec[-1].get("_id").generation_time
            # print(dt)
            if utc_now < dt + timedelta(
                days=7
            ):  # existing record < week old? if so, ignore it
                print("Ignoring {} as record is less than a week old.".format(asx_code))
                continue
        resp = fetcher.get(url, timeout=(30, 30))
        try:
            d = resp.json()
            d.update({"asx_code": asx_code})
            if "error_desc" in d:
                print(
                    "WARNING: ignoring {} ASX code as it is not a valid code".format(
                        asx_code
                    )
                )
                print(d)
            else:
                assert d.pop("code", None) == asx_code
                db.asx_company_details.delete_one({"asx_code": asx_code})
                db.asx_company_details.insert_one(
                    d
                )  # ensure new _id is assigned with current date
        except Exception as e:
            print(str(e))
            pass
        time.sleep(5)


def fix_blacklist(db, config):
    updates = {}
    # we generate a random
    for rec in db.asx_blacklist.find({}):
        future_date = datetime.today().date() + timedelta(days=randint(30, 90))
        updates[rec.get("asx_code")] = future_date

    n = 0
    for stock, future_date in updates.items():
        db.asx_blacklist.update_one(
            {"asx_code": stock},
            {
                "$set": {
                    "valid_until": datetime.combine(future_date, datetime.min.time())
                }
            },
        )
        n += 1
    print("Updated {} blacklist entries.".format(n))


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Fetch and update data from ASX Research"
    )
    args.add_argument(
        "--blacklist",
        help="Update blacklist table with dead stocks",
        action="store_true",
    )
    args.add_argument(
        "--config",
        help="Configuration file to use [config.json]",
        type=str,
        default="config.json",
    )
    args.add_argument(
        "--want-companies", help="Update companies list", action="store_true"
    )
    args.add_argument("--want-isin", help="Update securities list", action="store_true")
    args.add_argument(
        "--want-prices",
        help="Update ASX stock price list with current data",
        action="store_true",
    )
    args.add_argument(
        "--want-details",
        help="Update ASX company details (incl. dividend, annual report etc.) with current data",
        action="store_true",
    )
    args.add_argument("--validate", help="", action="store_true")
    args.add_argument(
        "--fix-blacklist",
        help="Ensure each blacklist entry has a valid_until date",
        action="store_true",
    )
    args.add_argument(
        "--date",
        help="Date to use as the record date in the database [YYYY-mm-dd]",
        type=str,
        required=False,
    )
    args.add_argument(
        "--stocks",
        help="JSON array with stocks to load for --want-prices",
        type=str,
        required=False,
    )
    a = args.parse_args()

    config, password = read_config(a.config)
    m = config.get("mongo")
    mongo = pymongo.MongoClient(
        m.get("host"), m.get("port"), username=m.get("user"), password=password
    )
    db = mongo[m.get("db")]

    if a.blacklist:
        update_blacklist(db)

    if a.want_companies:
        print("**** UPDATING ASX COMPANIES")
        update_companies(db, config, ensure_indexes=True)
    if a.want_isin:
        print("**** UPDATING ASX SECURITIES")
        update_isin(db, config, ensure_indexes=True)
    if a.fix_blacklist:
        print("*** FIX BLACKLIST ENTRIES")
        fix_blacklist(db, config)

    if any([a.want_prices, a.want_details]):
        if a.stocks:
            with open(a.stocks, "r") as fp:
                stocks_to_fetch = json.loads(fp.read())
        else:
            stocks_to_fetch = available_stocks(db, config)
        print("Found {} stocks to fetch.".format(len(stocks_to_fetch)))
        if a.want_prices:
            print("**** UPDATING PRICES")
            if a.date:
                import re

                pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
                assert pattern.match(a.date)
                fetch_date = a.date
            else:
                fetch_date = datetime.now().strftime("%Y-%m-%d")
            update_prices(db, stocks_to_fetch, config, fetch_date, ensure_indexes=True)
        if a.want_details:
            print("**** UPDATING COMPANY DETAILS")
            update_company_details(db, stocks_to_fetch, config, ensure_indexes=True)

    if a.validate:
        validate_prices("test.tsv")

    mongo.close()
    print("Run completed.")
    exit(0)
