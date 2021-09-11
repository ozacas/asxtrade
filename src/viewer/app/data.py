"""
Responsible for preparing pandas dataframes, will eventually become the basis for performing data downloads
from the app. A second responsibility for caching dynamic images via cache_plot()
"""
from collections import defaultdict
from datetime import datetime
import io
import re
import hashlib
from time import time
from typing import Any, Iterable, Callable
import math
import numpy as np
import pandas as pd
from cachetools import func
from lazydict import LazyDictionary
from app.models import (
    companies_with_same_sector,
    Timeframe,
    company_prices,
    validate_date,
    VirtualPurchase,
    timing,
)
from django.core.cache import cache
import plotnine as p9
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
from threading import Lock
from shapely.geometry import LineString, Point


def hash_str(key: str) -> str:
    assert len(key) > 0
    result = hashlib.sha256(key.encode("utf-8")).hexdigest()
    assert isinstance(result, str)
    return result


check_hash_collision_dict: dict = {}

# for the sole use by cache_plot() it is designed to ensure thread safety of the underlying matplotlib backend which is not designed for
# concurrent web-requests. Hopefully this is sufficient, but it may not be.
exclusive_lock = Lock()


@timing
def cache_plot(
    key: str,
    plot_factory: Callable = None,
    debug: bool = False,
    timeout: float = 120.0 * 60,
    datasets: LazyDictionary = None,
    django_cache=None,
    dont_cache: bool = False,
) -> str:
    """
    Using the specified key compute try to find a suitable dynamic image via diskcache. If not found plot_factory() is invoked
    to compute the image. Cache retrieval is disabled if dont_cache is True (useful for debugging). Dynamic images are cached for 120mins by default.
    """
    assert timeout > 0.0
    global exclusive_lock
    assert exclusive_lock is not None
    assert plot_factory is not None

    if django_cache is None:
        django_cache = cache

    try:
        exclusive_lock.acquire()

        # ensure all URLs are sha256 hexdigest's regardless of the key specified to avoid data leak and security from brute-forcing
        cache_key = hash_str(key)

        if debug:  # check for unwanted collisions?
            if (
                cache_key in check_hash_collision_dict
                and key != check_hash_collision_dict[cache_key]
            ):
                raise ValueError(
                    "*** HASH collision! expected {key} but found: {check_hash_collision_dict[cache_key]}"
                )
            elif cache_key not in check_hash_collision_dict:
                check_hash_collision_dict[cache_key] = key

        # plot already done and in cache?
        if cache_key in django_cache and not dont_cache:
            return cache_key

        # TODO FIXME... thread safety and proper locking for this code
        with io.BytesIO(bytearray(200 * 1024)) as buf:
            plot = plot_factory(datasets)
            if plot is None:
                django_cache.delete(cache_key)
                return None
            if isinstance(plot, p9.ggplot):
                fig = plot.draw()  # need to invoke matplotlib.savefig() so deref ...
            else:
                try:
                    fig = plot.gcf()
                except AttributeError:
                    fig = plot  # matplotlib figures dont have gcf callable, so we use savefig directly for them
            fig.savefig(buf, format="png")
            n_bytes = buf.seek(
                0, 1
            )  # find out how many bytes have been written above by a "zero" seek
            assert n_bytes > 0
            buf.seek(0)
            if debug:
                print(
                    f"Setting cache plot (timeout {timeout} sec.): {key} -> {cache_key} n_bytes={n_bytes}"
                )
            django_cache.set(
                cache_key, buf.read(n_bytes), timeout=timeout, read=True
            )  # timeout must be large enough to be served on the page...
            plt.close(fig)
            return cache_key
    finally:
        exclusive_lock.release()


# BUGGY TODO FIXME: doesnt handle multiple-metric view well and choice of significant digits sucks also
def label_shorten(labels: Iterable[float]) -> Iterable[str]:
    """
    For large numbers we report them removing all the zeros and replacing with billions/millions/trillions as appropriate.
    This is only done if all labels in the plot end with zeros and all are sufficiently long to have enough zeros. Otherwise labels are untouched.
    """
    # print(labels)
    str_labels = []
    non_zero_labels = []
    for v in labels:
        magnitude_v = abs(v)
        new_str = str(int(v)) if magnitude_v > 100000.0 else "{:f}".format(v)
        # trim trailing zeros so it doesnt confuse the code below?
        if re.match(r"^.*\.0+$", new_str):
            new_str = re.sub(r"\.0+$", "", new_str)
            # fallthru...

        str_labels.append(
            new_str
        )  # NB: some plots have negative value tick marks, so we must handle it
        if magnitude_v > 1e-10:
            non_zero_labels.append(new_str)

    # print(str_labels)
    # print(non_zero_labels)
    non_zero_labels_set = set(non_zero_labels)

    for units, short_label in [
        ("000000000000", "T."),
        ("000000000", "B."),
        ("000000", "M."),
    ]:
        found_cnt = 0
        for l in non_zero_labels:
            if l.endswith(units):
                found_cnt += 1
        if found_cnt == len(non_zero_labels):
            new_labels = []
            for old_label in str_labels:
                # print(old_label)
                # print(units)
                if old_label in non_zero_labels_set:
                    base_label = old_label[0 : -len(units)]
                    new_label = base_label + short_label
                    new_labels.append(new_label)
                else:
                    new_labels.append(old_label)
            return new_labels
    # no way to consistently shorten labels? ok, return supplied labels
    return labels


def make_portfolio_performance_dataframe(
    stocks: Iterable[str], timeframe: Timeframe, purchases: Iterable[VirtualPurchase]
) -> pd.DataFrame:
    def sum_portfolio(df: pd.DataFrame, date_str: str, stock_items):
        validate_date(date_str)

        portfolio_worth = sum(map(lambda t: df.at[t[0], date_str] * t[1], stock_items))
        return portfolio_worth

    df = company_prices(stocks, timeframe, transpose=True)
    rows = []
    stock_count = defaultdict(int)
    stock_cost = defaultdict(float)
    portfolio_cost = 0.0

    for d in [datetime.strptime(x, "%Y-%m-%d").date() for x in timeframe.all_dates()]:
        d_str = str(d)
        if d_str not in df.columns:  # not a trading day?
            continue
        purchases_to_date = filter(lambda vp, d=d: vp.buy_date <= d, purchases)
        for purchase in purchases_to_date:
            if purchase.buy_date == d:
                portfolio_cost += purchase.amount
                stock_count[purchase.asx_code] += purchase.n
                stock_cost[purchase.asx_code] += purchase.amount

        portfolio_worth = sum_portfolio(df, d_str, stock_count.items())
        # print(df)
        # emit rows for each stock and aggregate portfolio
        for asx_code in stocks:
            cur_price = df.at[asx_code, d_str]
            if np.isnan(cur_price):  # price missing? ok, skip record
                continue
            assert cur_price is not None and cur_price >= 0.0
            stock_worth = cur_price * stock_count[asx_code]

            rows.append(
                {
                    "portfolio_cost": portfolio_cost,
                    "portfolio_worth": portfolio_worth,
                    "portfolio_profit": portfolio_worth - portfolio_cost,
                    "stock_cost": stock_cost[asx_code],
                    "stock_worth": stock_worth,
                    "stock_profit": stock_worth - stock_cost[asx_code],
                    "date": d_str,
                    "stock": asx_code,
                }
            )

    df = pd.DataFrame.from_records(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def make_portfolio_dataframe(df: pd.DataFrame, melt=False):
    assert df is not None
    # print(df)
    df["date"] = pd.to_datetime(df["date"])
    avg_profit_over_period = (
        df.filter(items=["stock", "stock_profit"]).groupby("stock").mean()
    )
    avg_profit_over_period["contribution"] = [
        "positive" if profit >= 0.0 else "negative"
        for profit in avg_profit_over_period.stock_profit
    ]
    # dont want to override actual profit with average
    avg_profit_over_period = avg_profit_over_period.drop("stock_profit", axis="columns")
    df = df.merge(
        avg_profit_over_period, left_on="stock", right_index=True, how="inner"
    )
    # print(df)
    if melt:
        df = df.filter(
            items=["stock", "date", "stock_profit", "stock_worth", "contribution"]
        )
        melted_df = df.melt(id_vars=["date", "stock", "contribution"], var_name="field")
        return melted_df
    return df


def make_sector_performance_dataframe(
    all_stocks_cip: pd.DataFrame, sector_companies=None
) -> pd.DataFrame:
    cip = prep_cip_dataframe(all_stocks_cip, None, sector_companies)
    if cip is None:
        return None

    rows = []
    cum_sum = defaultdict(float)
    for day in sorted(cip.columns, key=lambda k: datetime.strptime(k, "%Y-%m-%d")):
        for asx_code, daily_change in cip[day].iteritems():
            cum_sum[asx_code] += daily_change
        n_pos = len(list(filter(lambda t: t[1] >= 5.0, cum_sum.items())))
        n_neg = len(list(filter(lambda t: t[1] < -5.0, cum_sum.items())))
        n_unchanged = len(cip) - n_pos - n_neg
        rows.append(
            {"n_pos": n_pos, "n_neg": n_neg, "n_unchanged": n_unchanged, "date": day}
        )

    df = pd.DataFrame.from_records(rows)
    return df


def prep_cip_dataframe(
    cip: pd.DataFrame, stock: str, sector_companies: Iterable[str] = None
) -> pd.DataFrame:
    assert (stock is None and sector_companies is not None) or (stock is not None)
    if sector_companies is None:
        sector_companies = companies_with_same_sector(stock)
    if len(sector_companies) == 0:
        return None

    cip = cip.filter(items=sector_companies, axis="index")
    cip = cip.fillna(0.0)
    return cip


@timing
def make_stock_vs_sector_dataframe(
    all_stocks_cip: pd.DataFrame, stock: str, sector_companies=None
) -> pd.DataFrame:
    # print(all_stocks_cip)
    cip = prep_cip_dataframe(all_stocks_cip, stock, sector_companies)
    if cip is None:
        return None

    # identify the best performing stock in the sector and add it to the stock_versus_sector rows...
    best_stock_in_sector = cip.sum(axis=1).nlargest(1).index[0]
    best_group = "{} (#1 in sector)".format(best_stock_in_sector)
    # ensure cip columns are in correct date order
    cip = cip.reindex(
        sorted(cip.columns, key=lambda k: datetime.strptime(k, "%Y-%m-%d")), axis=1
    )
    #     fetch_date  2019-08-23  2019-08-26  2019-08-27  2019-08-28  2019-08-29  2019-08-30  ...  2021-08-13  2021-08-16  2021-08-17  2021-08-18  2021-08-19  2021-08-20
    # asx_code                                                                            ...
    # NVU           0.000000    0.000000    0.000000    0.000000    0.000000    0.000000  ...    4.255000       2.041      -2.000       0.000       2.041      -8.000
    # 3DP           2.564103    0.000000    2.631579   -5.128205   -2.631579    5.263158  ...    2.469000       0.000      -1.220       0.000       1.235      -3.614
    # TDY           0.000000    0.000000    0.000000    0.000000    0.000000    0.000000  ...    0.000000       0.000       0.000       0.000       0.000       0.000
    # AMO           0.000000    0.000000    0.000000    0.000000    0.000000    0.000000  ...    1.961000       5.769       1.818      -1.786      -1.818       3.704
    # YOJ           0.000000   -1.428571    0.000000  -10.606061    0.000000   24.590164  ...    2.941000       0.000      -5.714       6.061      -5.714       0.000
    # ...                ...         ...         ...         ...         ...         ...  ...         ...         ...         ...         ...         ...         ...
    # ERD           0.000000    0.000000    0.000000    0.000000    0.000000    0.000000  ...    0.333333      -0.828      -1.169      -1.858       1.549       1.695
    # 9SP           8.695652   -3.846154   -3.846154   -8.000000   -4.347826    4.545455  ...   -6.667000       7.143       0.000       0.000       0.000       0.000
    # DDR           1.550388   -0.763359    1.846154    1.649175   -3.757225   -5.685131  ...   -1.400000       0.203      -3.509       0.490       1.809       2.051
    # BTH          -3.488372    2.469136    7.142857    1.098901    1.030928    0.000000  ...    0.000000      -5.000       0.000       0.877       4.783      -0.830
    # DUB          -0.803213   -0.823045    1.224490    4.780876   -3.030303   -5.859375  ...   -1.897000       0.829      -3.014       1.695       0.278      -2.216s

    cum_sum_df = cip.apply(lambda daily_prices: daily_prices.cumsum(), axis=1)
    sector_average = cum_sum_df.mean()  # arithmetic not weighted mean
    # print(sector_average)
    records = []
    for date in cip.columns:
        records.append(
            {"group": stock, "value": cum_sum_df.at[stock, date], "date": date}
        )
        records.append(
            {"group": "sector_average", "value": sector_average.at[date], "date": date}
        )
        if stock != best_stock_in_sector:
            records.append(
                {
                    "group": best_group,
                    "value": cum_sum_df.at[best_stock_in_sector, date],
                    "date": date,
                }
            )

    df = pd.DataFrame.from_records(records)
    return df


def make_point_score_dataframe(
    stock: str,
    rules: Iterable[tuple],
    ld: LazyDictionary,
) -> tuple:
    sector_companies = ld["sector_companies"]
    if len(sector_companies) < 1:
        return None, None

    rows = []
    points = 0
    day_low_high_df = ld["stock_df"]
    cip = ld["cip_df"]
    state = {
        "day_low_high_df": day_low_high_df,  # never changes each day, so we init it here
        "all_stocks_change_in_percent_df": cip,
        "stock": stock,
        "daily_range_threshold": 0.20,  # 20% at either end of the daily range gets a point
    }
    net_points_by_rule = defaultdict(int)
    for date in cip.columns:
        market_avg = cip[date].mean()
        sector_avg = cip[date].filter(items=sector_companies).mean()
        stock_move = cip.at[stock, date]
        state.update(
            {
                "market_avg": market_avg,
                "sector_avg": sector_avg,
                "stock_move": stock_move,
                "date": date,
            }
        )
        points += sum(map(lambda r: r(state), rules))
        for r in rules:
            k = r.__name__
            if k.startswith("rule_"):
                k = k[5:]
            net_points_by_rule[k] += r(state)
        rows.append({"points": points, "stock": stock, "date": date})

    df = pd.DataFrame.from_records(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df, net_points_by_rule


def make_kmeans_cluster_dataframe(
    timeframe: Timeframe, chosen_k: int, stocks: Iterable[str]
) -> tuple:
    prices_df = company_prices(stocks, timeframe, fields="last_price")
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(prices_df)
    s1 = prices_df.pct_change().mean() * 252
    s2 = prices_df.pct_change().std() * math.sqrt(252.0)
    # print(s1)
    data_df = pd.DataFrame.from_dict({"return": s1, "volatility": s2})
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(data_df)
    data_df = data_df.dropna()  # calculation may produce inf/nan so purge now...
    data = np.asarray(
        [np.asarray(data_df["return"]), np.asarray(data_df["volatility"])]
    ).T
    distortion = []
    for k in range(2, 20):
        k_means = KMeans(n_clusters=k)
        k_means.fit(data)
        distortion.append(k_means.inertia_)
    # computing K-Means with K = 5 (5 clusters)
    centroids, _ = kmeans(data, chosen_k)
    # assign each sample to a cluster
    idx, _ = vq(data, centroids)
    data_df["cluster_id"] = idx
    return distortion, chosen_k, centroids, idx, data_df


def price_change_bins() -> tuple:
    """
    Return the change_in_percent numeric bins and corresponding labels as a tuple-of-lists for sentiment heatmaps to use and the
    plotting code. These are non-uniform bins, designed to be fairly sensitive to major market moves.

    """
    bins = [
        -1000.0,
        -100.0,
        -10.0,
        -5.0,
        -3.0,
        -2.0,
        -1.0,
        -1e-6,
        0.0,
        1e-6,
        1.0,
        2.0,
        3.0,
        5.0,
        10.0,
        25.0,
        100.0,
        1000.0,
    ]
    labels = ["{}".format(b) for b in bins[1:]]
    return (bins, labels)


@func.lru_cache(maxsize=1)
def pe_trends_df(timeframe: Timeframe) -> pd.DataFrame:
    # we fetch all required fields for this view in one call to company_prices() - more efficient on DB
    df = company_prices(
        None, timeframe, fields=["pe", "eps", "number_of_shares"], missing_cb=None
    )
    return df


def make_pe_trends_eps_df(
    df: pd.DataFrame, stocks_with_sector: pd.DataFrame = None
) -> pd.DataFrame:
    eps_df = df[df["field_name"] == "eps"].pivot(
        index="asx_code", columns="fetch_date", values="field_value"
    )
    if stocks_with_sector is not None:
        eps_df = eps_df.merge(stocks_with_sector, left_index=True, right_on="asx_code")
    return eps_df


def make_pe_trends_positive_pe_df(
    df: pd.DataFrame, stocks_with_sector: pd.DataFrame
) -> tuple:
    pe_df = df[df["field_name"] == "pe"].pivot(
        index="asx_code", columns="fetch_date", values="field_value"
    )
    positive_pe_stocks = set(pe_df[pe_df.mean(axis=1) > 0.0].index)
    # print(positive_pe_stocks)
    # print(pe_df)
    # print(stocks_with_sector)
    pe_pos_df = pe_df.filter(items=positive_pe_stocks, axis=0).merge(
        stocks_with_sector, left_index=True, right_on="asx_code"
    )
    pe_pos_df = pe_pos_df.set_index("asx_code", drop=True)

    return pe_pos_df, positive_pe_stocks


def timeframe_end_performance(ld: LazyDictionary) -> pd.Series:
    if ld is None or "cip_df" not in ld:
        return None

    timeframe_end_perf = ld["cip_df"].sum(axis=1, numeric_only=True)
    # print(timeframe_end_perf)
    return timeframe_end_perf.to_dict()


def calc_ma_crossover_points(
    array1: pd.Series, array2: pd.Series
) -> Iterable[tuple]:  # array indexes are returned
    """
    Calculate the points (index, YYYY-mm-dd, price) where array1 and array2 cross over and return them.
    Typically they represent MA20 and MA200 lines, but any two poly lines will do.
    """
    assert len(array1) == len(array2)
    ls1 = []
    ls2 = []
    for i in range(len(array1)):
        v1 = array1[i]
        v2 = array2[i]
        if np.isnan(v1) or np.isnan(
            v2
        ):  # TODO FIXME: gaps can create false overlaps due to "straight line" intersections in error... GIGO...
            continue
        ls1.append((i, v1))
        ls2.append((i, v2))

    result = None
    ls1 = LineString(ls1)
    ls2 = LineString(ls2)
    if ls1.crosses(ls2):  # symettry: implies ls2.crosses(ls1)
        result = ls1.intersection(ls2)
    if result is None or result.is_empty:
        return []
    elif isinstance(result, Point):
        nearest_idx = int(result.x)
        return [(nearest_idx, array1[nearest_idx], result.y)]
    else:  # assume multiple points ie. MultiPoint
        ret = []
        # print(result)
        for hit in list(result):
            if isinstance(hit, Point):
                nearest_idx = int(hit.x)
                intersect_price = hit.y
            elif isinstance(hit, LineString):
                # if a hit exists over multiple consecutive days, we just report the start
                coords = list(hit.coords)
                nearest_idx, intersect_price = coords[0]
            else:
                assert False  # fail for obscure corner cases
            # print(array1)
            ret.append((nearest_idx, array1.index[nearest_idx], intersect_price))

        return ret
