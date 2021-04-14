"""
Responsible for handling requests for pages from the website and delegating the analysis
and visualisation as required.
"""
from numpy import isnan
import pandas as pd
from collections import defaultdict
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from app.models import (
    user_watchlist,
    latest_quotation_date,
    cached_all_stocks_cip,
    company_prices,
    Timeframe,
    validate_user,
    stocks_by_sector
)
from app.data import cache_plot
from app.plots import (
    cached_heatmap,
    plot_series,
    plot_market_wide_sector_performance,
    plot_market_cap_distribution,
    plot_sector_field,
)

@login_required
def market_sentiment(request, n_days=21, n_top_bottom=20, sector_n_days=180):
    validate_user(request.user)
    assert n_days > 0
    assert n_top_bottom > 0
    timeframe = Timeframe(past_n_days=n_days)
    sector_timeframe = Timeframe(past_n_days=sector_n_days)
    df = cached_all_stocks_cip(timeframe)
    sector_df = cached_all_stocks_cip(sector_timeframe)
    sentiment_plot = cached_heatmap(tuple(df.index), timeframe, df)
    sector_performance_plot = cache_plot(f"sector-performance-{sector_timeframe.description}", 
                                      lambda: plot_market_wide_sector_performance(sector_df))
    market_cap_dist_plot = cache_plot(f"market-cap-dist-{sector_timeframe.description}",
                                      lambda: plot_market_cap_distribution(tuple(df.index), latest_quotation_date('ANZ'), sector_df.columns[0]))
    context = {
        "sentiment_uri": sentiment_plot,
        "n_days": timeframe.n_days,
        "n_stocks_plotted": len(df),
        "n_top_bottom": n_top_bottom,
        #"best_ten": top10,
        #"worst_ten": bottom10,
        "watched": user_watchlist(request.user),
        "sector_performance_uri": sector_performance_plot,
        "sector_performance_title": "Cumulative sector avg. performance: {}".format(sector_timeframe.description),
        "title": "Market sentiment: {}".format(timeframe.description),
        "market_cap_distribution_uri": market_cap_dist_plot
    }
    return render(request, "market_sentiment_view.html", context=context)


@login_required
def show_pe_trends(request):
    """
    Display a plot of per-sector PE trends across stocks in each sector
    ref: https://www.commsec.com.au/education/learn/choosing-investments/what-is-price-to-earnings-pe-ratio.html
    """
    validate_user(request.user)
    timeframe = Timeframe(past_n_days=180)
    pe_df = company_prices(None, timeframe, fields="pe", missing_cb=None, transpose=True)
    eps_df = company_prices(None, timeframe, fields="eps", missing_cb=None, transpose=True)
    ss = stocks_by_sector() 
    ss_dict = {row.asx_code: row.sector_name for row in ss.itertuples()}
    #print(ss_dict)
    eps_stocks = set(eps_df.index)
    n_stocks = len(pe_df)
    positive_pe_stocks = set(pe_df[pe_df.sum(axis=1) > 0.0].index)
    all_stocks = set(pe_df.index)
    n_non_zero_sum = len(positive_pe_stocks)
    #print(exclude_zero_sum)
    records = []
    trading_dates = set(pe_df.columns)
   
    sector_counts_all_stocks = ss['sector_name'].value_counts()
    all_sectors = set(ss['sector_name'].unique())
    pe_pos_df = pe_df.filter(items=positive_pe_stocks, axis=0).merge(ss, left_index=True, right_on='asx_code')
    assert len(pe_pos_df) <= len(positive_pe_stocks) and len(pe_pos_df) > 0
    market_avg_pe_df = pe_pos_df.mean(axis=0).to_frame(name='market_pe') # avg P/E by date series
    market_avg_pe_df['date'] = pd.to_datetime(market_avg_pe_df.index)
    #print(market_avg_pe_df)
    breakdown_by_sector_pe_pos_stocks_only = pe_pos_df['sector_name'].value_counts()
    #print(breakdown_by_sector_pe_pos_stocks_only)
    sector_counts_pe_pos_stocks_only = {s[0]: s[1] for s in breakdown_by_sector_pe_pos_stocks_only.items()}
    #print(sector_counts_pe_pos_stocks_only)
    #print(sector_counts_all_stocks)
    #print(sector_counts_pe_pos_stocks_only)
    for ymd in filter(lambda d: d in trading_dates, timeframe.all_dates()):  # needed to avoid KeyError raised during DataFrame.at[] calls below
        sum_pe_per_sector = defaultdict(float)
        sum_eps_per_sector = defaultdict(float)

        for stock in filter(lambda code: code in ss_dict, all_stocks):
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

        #print(sum_pe_per_sector)
        assert len(sector_counts_pe_pos_stocks_only) == len(sum_pe_per_sector)
        assert len(sector_counts_all_stocks) == len(sum_eps_per_sector)
        for sector in all_sectors:
            pe_sum = sum_pe_per_sector.get(sector, None)
            n_pe = sector_counts_pe_pos_stocks_only.get(sector, None)
            pe_mean = pe_sum / n_pe if pe_sum is not None else None
            eps_sum = sum_eps_per_sector.get(sector, None)

            records.append({'date': ymd, 'sector': sector, 'mean_pe': pe_mean, 'sum_pe': pe_sum, 
                            'sum_eps': eps_sum, 'n_stocks': n_stocks, 'n_sector_stocks_pe_only': n_pe })
        
    df = pd.DataFrame.from_records(records)
    # these arent per-user plots: they can safely be shared across all users of the site, so the key reflects that
    sector_pe_cache_key = f"{timeframe.description}-by-sector-pe-plot"
    sector_eps_cache_key = f"{timeframe.description}-by-sector-eps-plot"
    market_pe_cache_key = f"{timeframe.description}-market-pe-mean"
    market_pe_plot_uri = cache_plot(market_pe_cache_key,
                                    lambda: plot_series(market_avg_pe_df, x='date', y='market_pe', 
                                                        y_axis_label="Market-wide mean P/E", 
                                                        color=None, use_smooth_line=True))
    #print(df[df["sector"] == 'Utilities'])
    #print(df)
    context = {
        "title": "PE Trends: {}".format(timeframe.description),
        "n_stocks": n_stocks,
        "timeframe": timeframe,
        "n_stocks_with_pe": n_non_zero_sum,
        "sector_pe_plot_uri": cache_plot(sector_pe_cache_key, lambda: plot_sector_field(df, field="mean_pe")),
        "sector_eps_plot_uri": cache_plot(sector_eps_cache_key, lambda: plot_sector_field(df, field="sum_eps")),
        "market_pe_plot_uri": market_pe_plot_uri
    }
    return render(request, "pe_trends.html", context)