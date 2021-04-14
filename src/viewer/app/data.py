"""
Responsible for preparing pandas dataframes, will eventually become the basis for performing data downloads
from the app. A second responsibility for caching dynamic images via cache_plot()
"""
from collections import defaultdict
from datetime import datetime
import io
import hashlib
from typing import Iterable
import pandas as pd
from app.models import companies_with_same_sector, day_low_high
from django.core.cache import cache
import plotnine as p9
import matplotlib.pyplot as plt


def hash_str(key: str) -> str:
    assert len(key) > 0
    result = hashlib.sha256(key.encode('utf-8')).hexdigest()
    assert isinstance(result, str)
    return result

def cache_plot(key: str, plot_factory=None, debug=False, timeout=1.0*60, django_cache=None):
    assert plot_factory is not None
    
    if django_cache is None:
        django_cache = cache

    # ensure all URLs are sha256 hexdigest's regardless of the key specified to avoid data leak and security from brute-forcing
    cache_key = hash_str(key)

    # plot already done and in cache?
    if cache_key in django_cache:
        return cache_key

    with io.BytesIO(bytearray(200*1024)) as buf:
        plot = plot_factory()
        if plot is None:
            django_cache.delete(cache_key)
            return None
        if isinstance(plot, p9.ggplot):
            fig = plot.draw() # need to invoke matplotlib.savefig() so deref ...
        else:
            try:
                fig = plot.gcf()
            except AttributeError:
                fig = plot # matplotlib figures dont have gcf callable, so we use savefig directly for them
        fig.savefig(buf, format='png')
        buf.seek(0)
        if debug:
            print(f"Setting cache plot (timeout {timeout} sec.): {key} -> {cache_key}")
        django_cache.set(cache_key, buf.read(), timeout=timeout, read=True) # no page should take 10 minutes to render so this should guarantee object exists when served...
        plt.close(fig)
        return cache_key

def make_portfolio_dataframe(df: pd.DataFrame, melt=False):
    assert df is not None
    #print(df)
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
    df = df.merge(avg_profit_over_period, left_on="stock", right_index=True, how="inner")
    # print(df)
    if melt:
        df = df.filter(items=["stock", "date", "stock_profit", "stock_worth", "contribution"])
        melted_df = df.melt(id_vars=["date", "stock", "contribution"], var_name="field")
        return melted_df
    return df

def make_sector_performance_dataframe(all_stocks_cip: pd.DataFrame, sector_companies=None) -> pd.DataFrame:
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
        rows.append({ 'n_pos': n_pos, 'n_neg': n_neg, 'n_unchanged': n_unchanged, 'date': day})
       
    df = pd.DataFrame.from_records(rows)
    return df


def prep_cip_dataframe(cip: pd.DataFrame, stock: str, sector_companies:Iterable[str]=None) -> pd.DataFrame:
    assert (stock is None and sector_companies is not None) or (stock is not None)
    if sector_companies is None:
        sector_companies = companies_with_same_sector(stock)
    if len(sector_companies) == 0:
        return None

    cip = cip.filter(items=sector_companies, axis='index')
    cip = cip.fillna(0.0)
    return cip

def make_stock_vs_sector_dataframe(all_stocks_cip: pd.DataFrame, stock: str, sector_companies=None) -> pd.DataFrame:
    cip = prep_cip_dataframe(all_stocks_cip, stock, sector_companies)
    if cip is None:
        return None

    cum_sum = defaultdict(float)
    stock_versus_sector = []
    # identify the best performing stock in the sector and add it to the stock_versus_sector rows...
    best_stock_in_sector = cip.sum(axis=1).nlargest(1).index[0]
    best_group = '{} (#1 in sector)'.format(best_stock_in_sector)
    for day in sorted(cip.columns, key=lambda k: datetime.strptime(k, "%Y-%m-%d")):
        for asx_code, daily_change in cip[day].iteritems():
            cum_sum[asx_code] += daily_change
  
        stock_versus_sector.append({ 'group': stock, 'date': day, 'value': cum_sum[stock] })
        stock_versus_sector.append({ 'group': 'sector_average', 'date': day, 'value': pd.Series(cum_sum).mean() })
        if stock != best_stock_in_sector:
            stock_versus_sector.append({ 'group': best_group, 'value': cum_sum[best_stock_in_sector], 'date': day})
    df = pd.DataFrame.from_records(stock_versus_sector)
    return df


def make_point_score_dataframe(stock: str, sector_companies: Iterable[str], cip: pd.DataFrame, rules: Iterable[tuple]) -> tuple:
    rows = []
    points = 0
    day_low_high_df = day_low_high(stock, all_dates=cip.columns)
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
