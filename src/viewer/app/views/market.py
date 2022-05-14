"""
Responsible for handling requests for pages from the website and delegating the analysis
and visualisation as required.
"""
from collections import defaultdict
from operator import pos
from typing import Iterable
from numpy import isnan
import pandas as pd
from cachetools import func
from django.shortcuts import render
from django.http import Http404
from django.contrib.auth.decorators import login_required
from app.messages import warning
from app.models import (
    timing,
    user_watchlist,
    cached_all_stocks_cip,
    valid_quotes_only,
    all_stocks,
    Timeframe,
    validate_user,
    stocks_by_sector,
)
from app.data import (
    cache_plot,
    make_pe_trends_eps_df,
    make_pe_trends_positive_pe_df,
    pe_trends_df,
)
from app.plots import (
    plot_heatmap,
    plot_series,
    plot_market_wide_sector_performance,
    plot_market_cap_distribution,
    plot_sector_field,
    plot_sector_top_eps_contributors,
    plot_sector_monthly_mean_returns,
)
from lazydict import LazyDictionary


def bin_market_cap(row):
    mc = row[
        0
    ]  # NB: expressed in millions $AUD already (see plot_market_cap_distribution() below)
    if mc < 2000:
        return "small"
    elif mc > 10000:
        return "large"
    elif mc is not None:
        return "med"
    else:
        return "NA"


def make_quote_df(quotes, asx_codes: Iterable[str], prefix: str):
    df = pd.DataFrame.from_dict(
        {
            q.asx_code: (q.market_cap / (1000 * 1000), q.last_price, q.number_of_shares)
            for q in quotes
            if q.market_cap is not None and q.asx_code in asx_codes
        },
        orient="index",
        columns=["market_cap", "last_price", "shares"],
    )
    if len(df) == 0:
        print(quotes)
        raise Http404(f"No data present in {len(quotes)} quotes.")
    df["bin"] = df.apply(bin_market_cap, axis=1)
    df["market"] = prefix
    return df


@login_required
def market_sentiment(request, n_days=21, n_top_bottom=20, sector_n_days=365):
    validate_user(request.user)
    assert n_days > 0
    assert n_top_bottom > 0

    def market_cap_data_factory(ld: LazyDictionary) -> pd.DataFrame:
        dates = ld["sector_timeframe"].all_dates()
        # print(dates)
        assert len(dates) > 90
        result_df = None
        adjusted_dates = []
        for the_date in [dates[0], dates[-1], dates[-30], dates[-90]]:
            print(f"Before valid_quotes_only for {the_date}")
            quotes, actual_trading_date = valid_quotes_only(
                the_date, ensure_date_has_data=True
            )
            print(f"After valid_quotes_only for {the_date}")
            print(f"Before make quotes {actual_trading_date}")
            print(len(quotes))
            df = make_quote_df(quotes, ld["asx_codes"], actual_trading_date)
            print("After make_quote_df")
            result_df = df if result_df is None else result_df.append(df)
            if the_date != actual_trading_date:
                adjusted_dates.append(the_date)

        if len(adjusted_dates) > 0:
            warning(
                request,
                "Some dates were not trading days, adjusted: {}".format(adjusted_dates),
            )
        return result_df

    ld = LazyDictionary()
    ld["asx_codes"] = lambda ld: all_stocks()
    ld["sector_timeframe"] = lambda ld: Timeframe(past_n_days=sector_n_days)
    ld["timeframe"] = lambda ld: Timeframe(past_n_days=n_days)
    ld["sector_df"] = lambda ld: cached_all_stocks_cip(ld["sector_timeframe"])
    ld["sector_cumsum_df"] = lambda ld: ld["sector_df"].cumsum(axis=1)
    ld["cip_df"] = lambda ld: ld["sector_df"].filter(
        items=ld["timeframe"].all_dates(), axis=1
    )
    ld["market_cap_df"] = lambda ld: market_cap_data_factory(ld)
    ld["stocks_by_sector"] = lambda ld: stocks_by_sector()

    sentiment_plot = cache_plot(
        f"market-sentiment-{ld['timeframe'].description}",
        lambda ld: plot_heatmap(ld["timeframe"], ld),
        datasets=ld,
    )
    sector_descr = ld["sector_timeframe"].description
    sector_performance_plot = cache_plot(
        f"sector-performance-{sector_descr}",
        lambda ld: plot_market_wide_sector_performance(ld),
        datasets=ld,
    )
    market_cap_dist_plot = cache_plot(
        f"market-cap-dist-{sector_descr}",
        lambda ld: plot_market_cap_distribution(ld),
        datasets=ld,
    )

    df = ld["sector_cumsum_df"].transpose()
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
    df = (
        df.resample(
            "BM",
        )
        .asfreq()
        .diff(periods=1)
    )
    ld["monthly_returns_by_stock"] = df
    # print(df)

    context = {
        "sentiment_uri": sentiment_plot,
        "n_days": ld["timeframe"].n_days,
        "n_stocks_plotted": len(ld["asx_codes"]),
        "n_top_bottom": n_top_bottom,
        "watched": user_watchlist(request.user),
        "sector_performance_uri": sector_performance_plot,
        "sector_timeframe": ld["sector_timeframe"],
        "sector_performance_title": "Cumulative sector avg. performance: {}".format(
            ld["sector_timeframe"].description
        ),
        "title": "Market sentiment",
        "market_cap_distribution_uri": market_cap_dist_plot,
        "monthly_sector_mean_returns": plot_sector_monthly_mean_returns(ld),
    }
    return render(request, "market_sentiment_view.html", context=context)


@login_required
def show_pe_trends(request):
    """
    Display a plot of per-sector PE trends across stocks in each sector
    ref: https://www.commsec.com.au/education/learn/choosing-investments/what-is-price-to-earnings-pe-ratio.html
    """
    validate_user(request.user)

    def make_pe_trends_market_avg_df(ld: LazyDictionary) -> pd.DataFrame:
        df = ld["data_df"]
        ss = ld["stocks_by_sector"]
        pe_pos_df, _ = make_pe_trends_positive_pe_df(df, ss)
        market_avg_pe_df = pe_pos_df.mean(axis=0, numeric_only=True).to_frame(
            name="market_pe"
        )  # avg P/E by date series
        market_avg_pe_df["date"] = pd.to_datetime(market_avg_pe_df.index)
        return market_avg_pe_df

    def sector_eps_data_factory(ld: LazyDictionary) -> pd.DataFrame:
        df = ld["data_df"]
        n_stocks = df["asx_code"].nunique()
        pe_df, positive_pe_stocks = ld["positive_pe_tuple"]
        eps_df = ld["eps_df"]
        ss = ld["stocks_by_sector"]

        # print(positive_pe_stocks)
        eps_stocks = set(eps_df.index)
        ss_dict = {row.asx_code: row.sector_name for row in ss.itertuples()}
        # print(ss_dict)

        trading_dates = set(pe_df.columns)
        trading_dates.remove("sector_name")
        sector_counts_all_stocks = ss["sector_name"].value_counts()
        all_sectors = set(ss["sector_name"].unique())
        breakdown_by_sector_pe_pos_stocks_only = pe_df["sector_name"].value_counts()
        # print(breakdown_by_sector_pe_pos_stocks_only)
        sector_counts_pe_pos_stocks_only = {
            s[0]: s[1] for s in breakdown_by_sector_pe_pos_stocks_only.items()
        }
        # print(sector_counts_pe_pos_stocks_only)
        # print(sector_counts_all_stocks)
        # print(sector_counts_pe_pos_stocks_only)
        records = []
        for ymd in filter(
            lambda d: d in trading_dates, ld["timeframe"].all_dates()
        ):  # needed to avoid KeyError raised during DataFrame.at[] calls below
            sum_pe_per_sector = defaultdict(float)
            sum_eps_per_sector = defaultdict(float)

            for stock in filter(lambda code: code in ss_dict, positive_pe_stocks):
                sector = ss_dict[stock]
                assert isinstance(sector, str)

                if stock in eps_stocks:
                    eps = eps_df.at[stock, ymd]
                    if isnan(eps):
                        continue
                    sum_eps_per_sector[sector] += eps

                if stock in positive_pe_stocks:
                    pe = pe_df.at[stock, ymd]
                    if isnan(pe):
                        continue
                    assert pe >= 0.0
                    sum_pe_per_sector[sector] += pe

            # print(len(sector_counts_all_stocks))
            # print(len(sum_eps_per_sector))
            assert len(sector_counts_pe_pos_stocks_only) >= len(sum_pe_per_sector)
            assert len(sector_counts_all_stocks) >= len(sum_eps_per_sector)
            for sector in all_sectors:
                pe_sum = sum_pe_per_sector.get(sector, None)
                n_pe = sector_counts_pe_pos_stocks_only.get(sector, None)
                pe_mean = pe_sum / n_pe if pe_sum is not None else None
                eps_sum = sum_eps_per_sector.get(sector, None)

                records.append(
                    {
                        "date": ymd,
                        "sector": sector,
                        "mean_pe": pe_mean,
                        "sum_pe": pe_sum,
                        "sum_eps": eps_sum,
                        "n_stocks": n_stocks,
                        "n_sector_stocks_pe_only": n_pe,
                    }
                )
        df = pd.DataFrame.from_records(records)
        # print(df[df["sector"] == 'Utilities'])
        # print(df)
        return df

    ld = LazyDictionary()
    ld["data_df"] = lambda ld: pe_trends_df(ld["timeframe"])
    ld["positive_pe_tuple"] = lambda ld: make_pe_trends_positive_pe_df(
        ld["data_df"], ld["stocks_by_sector"]
    )
    ld["market_avg_pe_df"] = lambda ld: make_pe_trends_market_avg_df(ld)
    ld["eps_df"] = lambda ld: make_pe_trends_eps_df(ld["data_df"])
    ld["sector_eps_df"] = lambda ld: sector_eps_data_factory(ld)
    ld["stocks_by_sector"] = stocks_by_sector()
    ld["timeframe"] = Timeframe(past_n_days=180)
    td = ld["timeframe"].description

    # these arent per-user plots: they can safely be shared across all users of the site, so the key reflects that
    sector_pe_cache_key = f"{td}-by-sector-pe-plot"
    sector_eps_cache_key = f"{td}-by-sector-eps-plot"
    market_pe_cache_key = f"{td}-market-pe-mean"
    market_pe_plot_uri = cache_plot(
        market_pe_cache_key,
        lambda ld: plot_series(
            ld["market_avg_pe_df"],
            x="date",
            y="market_pe",
            y_axis_label="Market-wide mean P/E",
            color=None,
            use_smooth_line=True,
        ),
        datasets=ld,
    )

    context = {
        "title": "PE Trends",
        "n_stocks": ld["data_df"]["asx_code"].nunique(),
        "timeframe": ld["timeframe"],
        "n_stocks_with_pe": len(ld["positive_pe_tuple"][1]),
        "sector_pe_plot_uri": cache_plot(
            sector_pe_cache_key,
            lambda ld: plot_sector_field(ld["sector_eps_df"], field="mean_pe"),
            datasets=ld,
        ),
        "sector_eps_plot_uri": cache_plot(
            sector_eps_cache_key,
            lambda ld: plot_sector_field(ld["sector_eps_df"], field="sum_eps"),
            datasets=ld,
        ),
        "market_pe_plot_uri": market_pe_plot_uri,
        "sector_positive_top_contributors_eps_uri": cache_plot(
            f"top-contributors-{sector_eps_cache_key}",
            lambda ld: plot_sector_top_eps_contributors(
                ld["eps_df"], ld["stocks_by_sector"]
            ),
            datasets=ld,
        ),
    }
    return render(request, "pe_trends.html", context)
