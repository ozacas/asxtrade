"""
Responsible for preparing pandas dataframes, will eventually become the basis for performing data downloads
from the app. A second responsibility for caching dynamic images via cache_plot()
"""
from collections import defaultdict
from datetime import datetime
import io
import re
import hashlib
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


def make_stock_vs_sector_dataframe(
    all_stocks_cip: pd.DataFrame, stock: str, sector_companies=None
) -> pd.DataFrame:
    # print(all_stocks_cip)
    cip = prep_cip_dataframe(all_stocks_cip, stock, sector_companies)
    if cip is None:
        return None

    cum_sum = defaultdict(float)
    stock_versus_sector = []
    # identify the best performing stock in the sector and add it to the stock_versus_sector rows...
    best_stock_in_sector = cip.sum(axis=1).nlargest(1).index[0]
    best_group = "{} (#1 in sector)".format(best_stock_in_sector)
    for day in sorted(cip.columns, key=lambda k: datetime.strptime(k, "%Y-%m-%d")):
        for asx_code, daily_change in cip[day].iteritems():
            cum_sum[asx_code] += daily_change

        stock_versus_sector.append(
            {"group": stock, "date": day, "value": cum_sum[stock]}
        )
        stock_versus_sector.append(
            {"group": "sector_average", "date": day, "value": pd.Series(cum_sum).mean()}
        )
        if stock != best_stock_in_sector:
            stock_versus_sector.append(
                {
                    "group": best_group,
                    "value": cum_sum[best_stock_in_sector],
                    "date": day,
                }
            )
    df = pd.DataFrame.from_records(stock_versus_sector)
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
