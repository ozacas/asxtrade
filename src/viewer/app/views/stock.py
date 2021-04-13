"""
Responsible for providing detiled views about a single stock and closely related views
"""
from collections import defaultdict
from datetime import datetime
from cachetools import LFUCache
import pandas as pd
import numpy as np
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from app.models import (validate_stock, validate_user, stock_info, cached_all_stocks_cip, companies_with_same_sector,
                        Timeframe, company_prices, rsi_data, user_watchlist, selected_cached_stocks_cip, 
                        validate_date, user_purchases, all_available_dates, timing)
from app.analysis import (default_point_score_rules, rank_cumulative_change, 
                        calculate_trends, make_sector_performance_dataframe, make_stock_vs_sector_dataframe)
from app.messages import warning
from app.plots import (plot_point_scores, plot_fundamentals, make_rsi_plot, plot_trend, 
                        cached_portfolio_performance, cached_company_rank, plot_sector_performance, plot_company_versus_sector)

image_cache = LFUCache(maxsize=10)

@timing
def make_stock_sector(timeframe: Timeframe, stock: str) -> dict:
    cip = cached_all_stocks_cip(timeframe)
    sector_companies = companies_with_same_sector(stock)
    sector = stock_info(stock).get('sector_name', '')

    # implement caching (in memory) at image level to avoid all data manipulation if at all possible
    tag = "sector-momentum-plot-{}-{}".format(sector, timeframe.description)
    if tag in image_cache:
        sector_momentum_plot = image_cache[tag]
    else:
        df = make_sector_performance_dataframe(cip, sector_companies)
        sector_momentum_plot = plot_sector_performance(df, sector, window_size=14) if df is not None else None
        image_cache[tag] = sector_momentum_plot
    tag = "c_vs_s_plot-{}-{}".format(stock, timeframe.description)
    if tag in image_cache:
        c_vs_s_plot = image_cache[tag]
    else:
        df = make_stock_vs_sector_dataframe(cip, stock, sector_companies)
        c_vs_s_plot = plot_company_versus_sector(df, stock, sector) if df is not None else None
        image_cache[tag] = c_vs_s_plot

    # invoke separate function to cache the calls when we can
    point_score_plot = net_rule_contributors_plot = None
    if len(sector_companies) > 0:
        point_score_plot, net_rule_contributors_plot = \
                plot_point_scores(stock,
                                  sector_companies,
                                  cip,
                                  default_point_score_rules())
    return {  
        "timeframe": timeframe,
        "sector_momentum": {
            "plot": sector_momentum_plot,
            "title": "{} sector stocks".format(sector),
        },
        "company_versus_sector": {
            "plot": c_vs_s_plot,
            "title": "Performance against sector",
        },
        "point_score": {
            "plot": point_score_plot,
            "title": "Points score due to price movements",
        },
        "net_contributors": {
            "plot": net_rule_contributors_plot,
            "title": "Contributions to point score by rule",
        }
    }

@timing
def make_fundamentals(timeframe: Timeframe, stock: str):
    df = company_prices(
        [stock],
        timeframe,
        fields=("eps", "volume", "last_price", "annual_dividend_yield", \
                "pe", "change_in_percent", "change_price", "market_cap", \
                "number_of_shares"),
        missing_cb=None
    )
    #print(df)
    df['change_in_percent_cumulative'] = df['change_in_percent'].cumsum() # nicer to display cumulative
    df = df.drop('change_in_percent', axis=1)
    return {
        "plot": plot_fundamentals(df, stock),
        "title": "Stock fundamentals: EPS, PE, DY etc.",
        "timeframe": timeframe,
    }

@login_required
def show_stock(request, stock=None, n_days=2 * 365):
    """
    Displays a view of a single stock via the template and associated state
    """
    validate_stock(stock)
    validate_user(request.user)

    timeframe = Timeframe(past_n_days=n_days+200) # add 200 days so MA 200 can initialise itself before the plotting starts...
    plot_timeframe = Timeframe(past_n_days=n_days) # for template
    stock_df = rsi_data(stock, timeframe) # may raise 404 if too little data available
    company_details = stock_info(stock, lambda msg: warning(request, msg))
    momentum_plot = make_rsi_plot(stock, stock_df)

    # plot the price over timeframe in monthly blocks
    prices = stock_df[['last_price']].transpose() # use list of columns to ensure pd.DataFrame not pd.Series
    prices = prices.filter(items=plot_timeframe.all_dates(), axis='columns') # drop any date in "warm up" period
    monthly_maximum_plot = plot_trend(prices, sample_period='M')

    # populate template and render HTML page with context
    context = {
        "asx_code": stock,
        "watched": user_watchlist(request.user),
        "timeframe": plot_timeframe,
        "information": company_details,
        "momentum": {
           "rsi_plot": momentum_plot,
           "monthly_highest_price": {
                "title": "Highest price each month",
                "plot": monthly_maximum_plot,
           }
        },
        "fundamentals": make_fundamentals(plot_timeframe, stock),
        "stock_vs_sector": make_stock_sector(plot_timeframe, stock)
    }
    return render(request, "stock_page.html", context=context)


@login_required
def show_trends(request):
    validate_user(request.user)
    stocks = user_watchlist(request.user)
    timeframe = Timeframe(past_n_days=300)
    cip = selected_cached_stocks_cip(stocks, timeframe)
    trends = calculate_trends(cip, stocks)
    #print(trends)
    # for now we only plot trending companies... too slow and unreadable to load the page otherwise!
    cip = rank_cumulative_change(cip.filter(trends.keys(), axis="index"), timeframe)
    #print(cip)
    trending_companies_plot = cached_company_rank(cip, f"{request.user.username}-watchlist-trends")

    context = {
        "watchlist_trends": trends,
        "timeframe": timeframe,
        "trending_companies_uri": trending_companies_plot,
        "trending_companies_plot_title": "Trending watchlist stocks (ranked): {}".format(timeframe.description),
    }
    return render(request, "watchlist-rank.html", context=context)


def sum_portfolio(df: pd.DataFrame, date_str: str, stock_items):
    validate_date(date_str)

    portfolio_worth = sum(map(lambda t: df.at[t[0], date_str] * t[1], stock_items))
    return portfolio_worth

@login_required
def show_purchase_performance(request):
    validate_user(request.user)

    purchase_buy_dates = []
    purchases = []
    stocks = []
    for stock, purchases_for_stock in user_purchases(request.user).items():
        stocks.append(stock)
        for purchase in purchases_for_stock:
            purchase_buy_dates.append(purchase.buy_date)
            purchases.append(purchase)

    purchase_buy_dates = sorted(purchase_buy_dates)
    # print("earliest {} latest {}".format(purchase_buy_dates[0], purchase_buy_dates[-1]))

    timeframe = Timeframe(from_date=str(purchase_buy_dates[0]), to_date=all_available_dates()[-1])
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
        #print(df)
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
    username = request.user.username
    portfolio_performance_uri, stock_performance_uri, contributors_uri = cached_portfolio_performance(df, username)
   
    context = {
        "title": "Portfolio performance",
        "portfolio_title": "Overall",
        "performance_uri": portfolio_performance_uri,
        "stock_title": "Stock",
        "stock_performance_uri": stock_performance_uri,
        "contributors_uri": contributors_uri,
    }
    return render(request, "portfolio_trends.html", context=context)
