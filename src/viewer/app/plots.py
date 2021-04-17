"""
Responsible for production of data visualisations and rendering this data as inline
base64 data for various django templates to use.
"""
import base64
import io
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Iterable, Callable
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import plotnine as p9
from mizani.formatters import date_format
from django.contrib.auth import get_user_model
from app.models import stocks_by_sector, Timeframe, valid_quotes_only, timing, selected_cached_stocks_cip, user_purchases, all_available_dates
from app.data import make_sector_performance_dataframe, make_stock_vs_sector_dataframe, make_portfolio_dataframe, cache_plot, make_portfolio_performance_dataframe

def price_change_bins():
    """
    Return the change_in_percent bins and their label as a tuple for heatmap_market() to use and the
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

def cached_heatmap(asx_codes: Iterable[str], timeframe: Timeframe, df: pd.DataFrame=None) -> str:
    key = "-".join(asx_codes) + timeframe.description + "-stocks-sentiment"
    def inner(asx_codes, timeframe, df):
        if df is None:
            df = selected_cached_stocks_cip(asx_codes, timeframe)
        return plot_heatmap(df, timeframe)
    return cache_plot(key, lambda: inner(asx_codes, timeframe, df))

def cached_breakdown(asx_codes, timeframe: Timeframe, df: pd.DataFrame=None):
    key = "-".join(asx_codes)+"-breakdown"
    def inner(asx_codes, timeframe, df):
        if df is None:
            df = selected_cached_stocks_cip(asx_codes, timeframe)
        return plot_breakdown(df)
    return cache_plot(key, lambda: inner(asx_codes, timeframe, df))

def cached_company_versus_sector(stock: str, sector: str, sector_companies, cip: pd.DataFrame) -> str:
    cache_key = f"{stock}-{sector}-company-versus-sector"
    def inner():
        df = make_stock_vs_sector_dataframe(cip, stock, sector_companies)
        return plot_company_versus_sector(df, stock, sector) if df is not None else None
    return cache_plot(cache_key, inner)

def cached_sector_performance(sector: str, sector_companies, cip: pd.DataFrame, window_size=14):
    cache_key = f"{sector}-sector-performance"
    def inner():
        df = make_sector_performance_dataframe(cip, sector_companies)
        return plot_sector_performance(df, sector, window_size=window_size) if df is not None else None
    return cache_plot(cache_key, inner)

def cached_portfolio_performance(user):
    assert isinstance(user, get_user_model())
    username = user.username
    overall_key = f"{username}-portfolio-performance"
    stock_key = f"{username}-stock-performance"
    contributors_key = f"{username}-contributor-performance"

    def data_factory(): # dont create the dataframe unless we have to - avoid exxpensive call!
        purchase_buy_dates = []
        purchases = []
        stocks = []

        for stock, purchases_for_stock in user_purchases(user).items():
            stocks.append(stock)
            for purchase in purchases_for_stock:
                purchase_buy_dates.append(purchase.buy_date)
                purchases.append(purchase)

        purchase_buy_dates = sorted(purchase_buy_dates)
        # print("earliest {} latest {}".format(purchase_buy_dates[0], purchase_buy_dates[-1]))
        timeframe = Timeframe(from_date=str(purchase_buy_dates[0]), to_date=all_available_dates()[-1])

        return make_portfolio_performance_dataframe(stocks, timeframe, purchases)
    
    return (cache_plot(overall_key, lambda: plot_overall_portfolio(data_factory)),
            cache_plot(stock_key, lambda: plot_portfolio_stock_performance(data_factory)),
            cache_plot(contributors_key, lambda: plot_portfolio_contributors(data_factory)))
    
def make_sentiment_plot(sentiment_df, exclude_zero_bin=True, plot_text_labels=True):
    rows = []
    print(
        "Sentiment plot: exclude zero bins? {} show text? {}".format(
            exclude_zero_bin, plot_text_labels
        )
    )

    for column in filter(lambda c: c.startswith("bin_"), sentiment_df.columns):
        c = Counter(sentiment_df[column])
        date = column[4:]
        for bin_name, val in c.items():
            if exclude_zero_bin and (bin_name == "0.0" or not isinstance(bin_name, str)):
                continue
            bin_name = str(bin_name)
            assert isinstance(bin_name, str)
            val = int(val)
            rows.append(
                {
                    "date": datetime.strptime(date, "%Y-%m-%d"),
                    "bin": bin_name,
                    "value": val,
                }
            )

    df = pd.DataFrame.from_records(rows)
    # print(df['bin'].unique())
    bins, labels = price_change_bins() # pylint: disable=unused-variable
    order = filter(lambda s: s != "0.0", labels)  # dont show the no change bin since it dominates the activity heatmap
    df["bin_ordered"] = pd.Categorical(df["bin"], categories=order)

    plot = (
        p9.ggplot(df, p9.aes("date", "bin_ordered", fill="value"))
        + p9.geom_tile(show_legend=False)
        + p9.theme_bw()
        + p9.xlab("")
        + p9.ylab("Daily change (%)")
        + p9.theme(axis_text_x=p9.element_text(angle=30, size=7), figure_size=(10, 5))
    )
    if plot_text_labels:
        plot = plot + p9.geom_text(p9.aes(label="value"), size=8, color="white")
    return plot

def plot_fundamentals(df: pd.DataFrame, stock: str, line_size=1.5) -> str: # pylint: disable=unused-argument
    columns_to_report = ["pe", "eps", "annual_dividend_yield", "volume", \
                    "last_price", "change_in_percent_cumulative", \
                    "change_price", "market_cap", "number_of_shares"]
    colnames = df.columns
    for column in columns_to_report:
        assert column in colnames
   
    plot_df = pd.melt(
        df,
        id_vars="fetch_date",
        value_vars=columns_to_report,
        var_name="indicator",
        value_name="value",
    )
    plot_df["value"] = pd.to_numeric(plot_df["value"])
    n = len(columns_to_report)
    plot = (
        p9.ggplot(plot_df, p9.aes("fetch_date", "value", group="indicator", colour='indicator'))
        + p9.geom_path(show_legend=False, size=line_size)
        + p9.facet_wrap("~ indicator", nrow=n, ncol=1, scales="free_y")
        + p9.theme(axis_text_x=p9.element_text(angle=30, size=7), 
                   axis_text_y=p9.element_text(size=7),
                   figure_size=(8, n))
        + p9.scale_color_cmap_d()
        #    + p9.aes(ymin=0)
        + p9.xlab("")
        + p9.ylab("")
    )
    return plot


def plot_overall_portfolio(data_factory:Callable[[], pd.DataFrame], figure_size=(12, 4), line_size=1.5, date_text_size=7) -> p9.ggplot:
    """
    Given a daily snapshot of virtual purchases plot both overall and per-stock
    performance. Return a tuple of figures representing the performance as inline data.
    """
    portfolio_df = data_factory()

    df = portfolio_df.filter(
        items=["portfolio_cost", "portfolio_worth", "portfolio_profit", "date"]
    )
    df = df.melt(id_vars=["date"], var_name="field")
    plot = (
        p9.ggplot(df, p9.aes("date", "value", group="field", color="field"))
        + p9.labs(x="", y="$ AUD")
        + p9.geom_line(size=line_size)
        + p9.facet_wrap("~ field", nrow=3, ncol=1, scales="free_y")
        + p9.scale_colour_cmap_d()
        + p9.theme(
            axis_text_x=p9.element_text(angle=30, size=date_text_size),
            figure_size=figure_size,
            legend_position="none",
        )
    )
    return plot

def plot_portfolio_contributors(data_factory: Callable[[], pd.DataFrame], figure_size=(11,5)) -> p9.ggplot:
    df = data_factory()
    melted_df = make_portfolio_dataframe(df, melt=True)
    
    all_dates = sorted(melted_df["date"].unique())
    df = melted_df[melted_df["date"] == all_dates[-1]]
    df = df[df["field"] == "stock_profit"]  # only latest profit is plotted
    df["contribution"] = [
        "positive" if profit >= 0.0 else "negative" for profit in df["value"]
    ]

    # 2. plot contributors ie. winners and losers
    plot = (
        p9.ggplot(df, p9.aes("stock", "value", group="stock", fill="stock"))
        + p9.geom_bar(stat="identity")
        + p9.scale_fill_cmap_d()
        + p9.labs(x="", y="$ AUD")
        + p9.facet_grid("contribution ~ field", scales="free_y")
        + p9.theme(legend_position="none", figure_size=figure_size)
    )
    return plot

def plot_portfolio_stock_performance(data_factory: Callable[[], pd.DataFrame], figure_size=(11,5), date_text_size=7) -> p9.ggplot:
    def inner(pos_or_neg_val: float) -> float:
        val = abs(pos_or_neg_val)
        if val > 10000.0: # TODO FIXME: these numbers dont take into account the total portfolio cost... or purchase size
            return 1.0
        elif val > 3000.0:
            return 0.7
        elif val < 500.0: # stocks which a breakeven are very transparent
            return 0.2
        else:
            return 0.5

    df = data_factory()
    latest_date = df.iloc[-1, 6]
    latest_profit = df[df['date'] == latest_date]
    alpha_by_stock = defaultdict(float)
    for row in latest_profit.itertuples():
        alpha_by_stock[row.stock] = inner(row.stock_profit)

    melted_df = make_portfolio_dataframe(df, melt=True)
    melted_df['alpha'] = melted_df['stock'].apply(lambda stock: alpha_by_stock.get(stock, 1.0))
    plot = (
        p9.ggplot(melted_df, p9.aes("date", "value", group="stock", colour="stock"))
        + p9.xlab("")
        + p9.geom_line(size=1.0)
        + p9.scale_alpha(p9.aes(alpha='alpha'), guide=False)
        + p9.facet_grid("field ~ contribution", scales="free_y")
        + p9.scale_colour_cmap_d()
        + p9.theme(
            axis_text_x=p9.element_text(angle=30, size=date_text_size),
            figure_size=figure_size,
            panel_spacing=0.5,  # more space between plots to avoid tick mark overlap
            subplots_adjust={"right": 0.8},
        )
    )
    return plot


def plot_company_rank(data_factory: Callable[[], tuple]) -> p9.ggplot:
    df, _ = data_factory()  # trends from data_factory is ignored for this call, but the view needs it later...
    # assert 'sector' in df.columns
    n_bin = len(df["bin"].unique())
    #print(df)
    plot = (
        p9.ggplot(df, p9.aes("date", "rank", group="asx_code", color="asx_code"))
        + p9.geom_smooth(span=0.3, se=False)
        + p9.geom_text(
            p9.aes(label="asx_code", x="x", y="y"),
            nudge_x=1.2,
            size=6,
            show_legend=False,
        )
        + p9.xlab("")
        + p9.facet_wrap("~bin", nrow=n_bin, ncol=1, scales="free_y")
        + p9.theme(
            axis_text_x=p9.element_text(angle=30, size=7),
            figure_size=(8, 20),
            subplots_adjust={"right": 0.8},
        )
    )
    return plot


def plot_company_versus_sector(df: pd.DataFrame, stock: str, sector: str) -> str: # pylint: disable=unused-argument
    df["date"] = pd.to_datetime(df["date"])
    # print(df)
    plot = (
        p9.ggplot(
            df, p9.aes("date", "value", group="group", color="group", fill="group")
        )
        + p9.geom_line(size=1.5)
        + p9.xlab("")
        + p9.ylab("Change since start (%)")
        + p9.theme(
            axis_text_x=p9.element_text(angle=30, size=7),
            figure_size=(8, 4),
            subplots_adjust={"right": 0.8},
        )
    )
    return plot


def plot_market_wide_sector_performance(all_stocks_cip: pd.DataFrame):
    """
    Display specified dates for average sector performance. Each company is assumed to have at zero
    at the start of the observation period. A plot as base64 data is returned.
    """
    n_stocks = len(all_stocks_cip)
    # merge in sector information for each company
    code_and_sector = stocks_by_sector()
    n_unique_sectors = len(code_and_sector["sector_name"].unique())
    print("Found {} unique sectors".format(n_unique_sectors))

    #print(df)
    #print(code_and_sector)
    df = all_stocks_cip.merge(code_and_sector, left_index=True, right_on="asx_code")
    print(
        "Found {} stocks, {} sectors and merged total: {}".format(
            n_stocks, len(code_and_sector), len(df)
        )
    )
    # compute average change in percent of each unique sector over each day and sum over the dates
    cumulative_pct_change = df.expanding(axis="columns").sum()
    # merge date-wise into df
    for date in cumulative_pct_change.columns:
        df[date] = cumulative_pct_change[date]
    # df.to_csv('/tmp/crap.csv')
    grouped_df = df.groupby("sector_name").mean()
    # grouped_df.to_csv('/tmp/crap.csv')

    # ready the dataframe for plotting
    grouped_df = pd.melt(
        grouped_df,
        ignore_index=False,
        var_name="date",
        value_name="cumulative_change_percent",
    )
    grouped_df["sector"] = grouped_df.index
    grouped_df["date"] = pd.to_datetime(grouped_df["date"])
    n_col = 3
    plot = (
        p9.ggplot(
            grouped_df, p9.aes("date", "cumulative_change_percent", color="sector")
        )
        + p9.geom_line(size=1.5)
        + p9.facet_wrap(
            "~sector", nrow=n_unique_sectors // n_col + 1, ncol=n_col, scales="free_y"
        )
        + p9.xlab("")
        + p9.ylab("Average sector change (%)")
        + p9.scale_colour_cmap_d()
        + p9.theme(
            axis_text_x=p9.element_text(angle=30, size=6),
            axis_text_y=p9.element_text(size=6),
            figure_size=(12, 6),
            panel_spacing=0.3,
            legend_position="none",
        )
    )
    return plot


def plot_series(
        df,
        x=None,
        y=None,
        tick_text_size=6,
        line_size=1.5,
        y_axis_label="Point score",
        x_axis_label="",
        color="stock",
        use_smooth_line=False
):
    assert len(df) > 0
    assert len(x) > 0 and len(y) > 0
    assert line_size > 0.0
    assert isinstance(tick_text_size, int) and tick_text_size > 0
    assert y_axis_label is not None
    assert x_axis_label is not None
    args = {'x': x, 'y': y}
    if color:
        args['color'] = color
    plot = p9.ggplot(df, p9.aes(**args)) \
        + p9.labs(x=x_axis_label, y=y_axis_label) \
        + p9.theme(
            axis_text_x=p9.element_text(angle=30, size=tick_text_size),
            axis_text_y=p9.element_text(size=tick_text_size),
            legend_position="none",
        )
    if use_smooth_line:
        plot += p9.geom_smooth(size=line_size)
    else:
        plot += p9.geom_line(size=line_size)
    return plot

def bin_market_cap(row):
    mc = row[0] # NB: expressed in millions $AUD already (see plot_market_cap_distribution() below)
    if mc < 2000:
        return "small"
    elif mc > 10000:
        return "large"
    elif mc is not None:
        return "med"
    else:
        return "NA"

def make_quote_df(quotes, asx_codes, prefix):
    df = pd.DataFrame.from_dict({q.asx_code: (q.market_cap / (1000 * 1000), q.last_price, q.number_of_shares) 
                                for q in quotes if q.market_cap is not None and q.asx_code in asx_codes}, 
                                orient="index", columns=["market_cap", "last_price", "shares"])
    df['bin'] = df.apply(bin_market_cap, axis=1)
    df['market'] = prefix
    return df

def plot_market_cap_distribution(stocks, ymd: str, ymd_start_of_timeframe: str):
    #print(ymd)
    latest_quotes = valid_quotes_only(ymd)
    earliest_quotes = valid_quotes_only(ymd_start_of_timeframe)
    asx_codes = set(stocks)
   
    latest_df = make_quote_df(latest_quotes, asx_codes, ymd)
    earliest_df = make_quote_df(earliest_quotes, asx_codes, ymd_start_of_timeframe)
    df = latest_df.append(earliest_df)

    #print(df)
    small_text = p9.element_text(size=7)
    plot = p9.ggplot(df) + \
           p9.geom_boxplot(p9.aes(x='market', y='market_cap')) + \
           p9.facet_wrap("bin", scales="free_y") + \
           p9.labs(x='', y='Market cap. ($AUD Millions)') + \
           p9.theme(subplots_adjust={'wspace': 0.30}, 
                    axis_text_x=small_text, 
                    axis_text_y=small_text)
    return plot

def plot_breakdown(cip_df: pd.DataFrame):
    """Stacked bar plot of increasing and decreasing stocks per sector in the specified df"""
    cols_to_drop = [colname for colname in cip_df.columns if colname.startswith('bin_')]
    df = cip_df.drop(columns=cols_to_drop)
    df = pd.DataFrame(df.sum(axis='columns'), columns=['sum'])
    df = df.merge(stocks_by_sector(), left_index=True, right_on='asx_code')

    if len(df) == 0: # no stock in cip_df have a sector? ie. ETF?
        return None

    assert set(df.columns) == set(['sum', 'asx_code', 'sector_name'])
    df['increasing'] = df.apply(lambda row: 'up' if row['sum'] >= 0.0 else 'down', axis=1)
    sector_names = df['sector_name'].value_counts().index.tolist() # sort bars by value count (ascending)
    sector_names_cat = pd.Categorical(df['sector_name'], categories=sector_names)
    df = df.assign(sector_name_cat=sector_names_cat)

    #print(df)
    plot = (
        p9.ggplot(df, p9.aes(x='factor(sector_name_cat)', fill='factor(increasing)'))
        + p9.geom_bar()
        + p9.labs(x="Sector", y="Number of stocks")
        + p9.theme(axis_text_y=p9.element_text(size=7), 
                   subplots_adjust={"left": 0.2, 'right': 0.85},
                   legend_title=p9.element_blank()
                  )
        + p9.coord_flip()
    )
    return plot

def plot_heatmap(
        df: pd.DataFrame,
        timeframe: Timeframe,
        bin_cb=price_change_bins,
) -> p9.ggplot:
    """
    Plot the specified data matrix as binned values (heatmap) with X axis being dates over the specified timeframe and Y axis being
    the percentage change on the specified date (other metrics may also be used, but you will likely need to adjust the bins)
    Also computes top10/worst10 and returns a tuple (plot, top10, bottom10, n_stocks). Top10/Bottom10 will contain n_top_bottom stocks.
    """
    bins, labels = bin_cb()
    # print(df.columns)
    # print(bins)
    try:
        # NB: this may fail if no prices are available so we catch that error and handle accordingly...
        for date in df.columns:
            df["bin_{}".format(date)] = pd.cut(df[date], bins, labels=labels)
        sentiment_plot = make_sentiment_plot(df, plot_text_labels=timeframe.n_days <= 30)  # show counts per bin iff not too many bins
        return sentiment_plot
    except KeyError:
        return None


def plot_sector_performance(dataframe: pd.DataFrame, descriptor: str, window_size=14):
    assert len(dataframe) > 0

    fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
    timeline = pd.to_datetime(dataframe["date"])
    locator, formatter = auto_dates()
    # now do the plot
    for name, ax, linecolour, title in zip(
        ["n_pos", "n_neg", "n_unchanged"],
        axes,
        ["darkgreen", "red", "grey"],
        [
            "{} stocks up >5%".format(descriptor),
            "{} stocks down >5%".format(descriptor),
            "Remaining stocks",
        ],
    ):
        # use a moving average to smooth out 5-day trading weeks and see the trend
        series = dataframe[name].rolling(window_size).mean()
        ax.plot(timeline, series, color=linecolour)
        ax.set_ylabel("", fontsize=8)
        ax.set_ylim(0, max(series.fillna(0)) + 10)
        ax.set_title(title, fontsize=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Remove the automatic x-axis label from all but the bottom subplot
        if ax != axes[-1]:
            ax.set_xlabel("")
    return fig

def auto_dates():
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = [
        "%y",  # ticks are mostly years
        "%b",  # ticks are mostly months
        "%d",  # ticks are mostly days
        "%H:%M",  # hrs
        "%H:%M",  # min
        "%S.%f",
    ]  # secs
    # these are mostly just the level above...
    formatter.zero_formats = [""] + formatter.formats[:-1]
    # ...except for ticks that are mostly hours, then it is nice to have
    # month-day:
    formatter.zero_formats[3] = "%d-%b"

    formatter.offset_formats = [
        "",
        "%Y",
        "%b %Y",
        "%d %b %Y",
        "%d %b %Y",
        "%d %b %Y %H:%M",
    ]
    return (locator, formatter)


def relative_strength(prices, n=14):
    # see https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
    assert n > 0
    assert prices is not None

    # Get the difference in price from previous step
    delta = prices.diff()

    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = delta[1:]

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = up.ewm(span=n).mean()
    roll_down1 = down.abs().ewm(span=n).mean()

    # Calculate the RSI based on EWMA
    rs = roll_up1 / roll_down1
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # NB: format is carefully handled here, so downstream code doesnt break
    new_date = datetime.strftime(
        datetime.now(), "%Y-%m-%d "
    )  # make sure it is not an existing date
    # print(new_date)
    rsi.at[new_date] = np.nan  # ensure data series are the same length for matplotlib
    # print(len(rsi), " ", len(prices))
    # assert len(rsi) == len(prices)
    return rsi


def plot_momentum(data_factory: Callable[[], tuple], stock: str) -> plt.Figure:
    assert len(stock) > 0
    stock_df, prices = data_factory()

    # print(last_price)
    # print(volume)
    # print(day_low_price)
    # print(day_high_price)

    last_price = stock_df["last_price"]
    volume = stock_df["volume"]
    day_low_price = stock_df["day_low_price"]
    day_high_price = stock_df["day_high_price"]

    plt.rc("axes", grid=True)
    plt.rc("grid", color="0.75", linestyle="-", linewidth=0.5)

    textsize = 8
    left, width = 0.1, 0.8
    rect1 = [left, 0.7, width, 0.2]
    rect2 = [left, 0.3, width, 0.4]
    rect3 = [left, 0.1, width, 0.2]

    fig = plt.figure(facecolor="white", figsize=(12, 6))
    axescolor = "#f6f6f6"  # the axes background color

    ax1 = fig.add_axes(rect1, facecolor=axescolor)  # left, bottom, width, height
    ax2 = fig.add_axes(rect2, facecolor=axescolor, sharex=ax1)
    ax2t = ax2.twinx()
    ax3 = fig.add_axes(rect3, facecolor=axescolor, sharex=ax1)
    fig.autofmt_xdate()

    # plot the relative strength indicator
    rsi = relative_strength(last_price)
    # print(len(rsi))
    fillcolor = "darkgoldenrod"

    timeline = pd.to_datetime(last_price.index)
    # print(values)
    ax1.plot(timeline, rsi, color=fillcolor)
    ax1.axhline(70, color="darkgreen")
    ax1.axhline(30, color="darkgreen")
    ax1.fill_between(
        timeline, rsi, 70, where=(rsi >= 70), facecolor=fillcolor, edgecolor=fillcolor
    )
    ax1.fill_between(
        timeline, rsi, 30, where=(rsi <= 30), facecolor=fillcolor, edgecolor=fillcolor
    )
    ax1.text(
        0.6,
        0.9,
        ">70 = overbought",
        va="top",
        transform=ax1.transAxes,
        fontsize=textsize,
    )
    ax1.text(0.6, 0.1, "<30 = oversold", transform=ax1.transAxes, fontsize=textsize)
    ax1.set_ylim(0, 100)
    ax1.set_yticks([30, 70])
    ax1.text(
        0.025, 0.95, "RSI (14)", va="top", transform=ax1.transAxes, fontsize=textsize
    )
    # ax1.set_title('{} daily'.format(stock))

    # plot the price and volume data
    dx = 0.0
    low = day_low_price + dx
    high = day_high_price + dx

    deltas = np.zeros_like(last_price)
    deltas[1:] = np.diff(last_price)
    up = deltas > 0
    ax2.vlines(timeline[up], low[up], high[up], color="black", label="_nolegend_")
    ax2.vlines(timeline[~up], low[~up], high[~up], color="black", label="_nolegend_")
    ma20 = last_price.rolling(window=20).mean()
    ma200 = last_price.rolling(window=200).mean()

    # timeline = timeline.to_list()
    (linema20,) = ax2.plot(timeline, ma20, color="blue", lw=2, label="MA (20)")
    (linema200,) = ax2.plot(timeline, ma200, color="red", lw=2, label="MA (200)")
    assert linema20 is not None
    assert linema200 is not None
   
    props = font_manager.FontProperties(size=10)
    leg = ax2.legend(loc="center left", shadow=True, fancybox=True, prop=props)
    leg.get_frame().set_alpha(0.5)

    volume = (last_price * volume) / 1e6  # dollar volume in millions
    # print(volume)
    vmax = max(volume)
    poly = ax2t.fill_between(
        timeline,
        volume.to_list(),
        0,
        alpha=0.5,
        label="Volume",
        facecolor=fillcolor,
        edgecolor=fillcolor,
    )
    assert poly is not None  # avoid unused variable from pylint
    ax2t.set_ylim(0, 5 * vmax)
    ax2t.set_yticks([])

    # compute the MACD indicator
    fillcolor = "darkslategrey"

    n_fast = 12
    n_slow = 26
    n_ema = 9
    emafast = last_price.ewm(span=n_fast, adjust=False).mean()
    emaslow = last_price.ewm(span=n_slow, adjust=False).mean()
    macd = emafast - emaslow
    nema = macd.ewm(span=n_ema, adjust=False).mean()
    ax3.plot(timeline, macd, color="black", lw=2)
    ax3.plot(timeline, nema, color="blue", lw=1)
    ax3.fill_between(
        timeline, macd - nema, 0, alpha=0.3, facecolor=fillcolor, edgecolor=fillcolor
    )
    ax3.text(
        0.025,
        0.95,
        "MACD ({}, {}, {})".format(n_fast, n_slow, n_ema),
        va="top",
        transform=ax3.transAxes,
        fontsize=textsize,
    )

    ax3.set_yticks([])
    locator, formatter = auto_dates()
    for ax in ax1, ax2, ax2t, ax3:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    plt.xticks(fontsize=8)
    try:
        plt.xlim(left=timeline[200])
    except IndexError:
        print("WARNING: 200 datapoints not available - some momentum data not available")
    plt.plot()
    fig = plt.gcf()
    plt.close(fig)
    return fig


def plot_trend(data_factory: Callable[[], tuple], sample_period="M") -> str:
    """
    Given a dataframe of a single stock from company_prices() this plots the highest price
    in each month over the time period of the dataframe.
    """
    def inner_date_fmt(dates_to_format):
        results = []
        for d in dates_to_format:
            d -= timedelta(weeks=4) # breaks are set to the end of the month rather than the start... so
            results.append(d.strftime("%Y-%m"))
        return results

    _, dataframe = data_factory()
    assert dataframe is not None
    dataframe = dataframe.transpose()
    dataframe.index = pd.to_datetime(dataframe.index, format='%Y-%m-%d')
    dataframe = dataframe.resample(sample_period).max()
    #print(dataframe.index)
    plot = (
        p9.ggplot(dataframe, p9.aes(x="dataframe.index", y=dataframe.columns[0]))
        + p9.geom_bar(stat="identity", fill="#880000", alpha=0.5)
        + p9.scale_x_datetime(labels=inner_date_fmt)     # dont print day (always 1st day of month due to resampling)
        + p9.labs(x="", y="$AUD")
        + p9.theme(axis_text_x=p9.element_text(angle=30, size=7))
    )
    return plot

def plot_point_scores(cache_key: str, point_score_dataframe: pd.DataFrame) -> str:
    """
    Visualise the stock in terms of point scores as described on the stock view page. 
    :param: point_score_dataframe result from call to make_point_score_dataframe() ie. ready to plot DataFrame
    :rtype: string for accessing the plot via the Django cache
    """
    return cache_plot(cache_key, lambda: plot_series(point_score_dataframe, x="date", y="points"))

def plot_points_by_rule(cache_key: str, net_points_by_rule: defaultdict(int)) -> str:
    def inner():
        rows = []
        for k, v in net_points_by_rule.items():
            rows.append({"rule": str(k), "net_points": v})
        df = pd.DataFrame.from_records(rows)
        return (
            p9.ggplot(df, p9.aes(x="rule", y="net_points"))
            + p9.labs(x="Rule", y="Contribution to points by rule")
            + p9.geom_bar(stat="identity", fill="#880000", alpha=0.5)
            + p9.theme(axis_text_y=p9.element_text(size=7), subplots_adjust={"left": 0.2})
            + p9.coord_flip()
        )
    return cache_plot(cache_key, inner)


def plot_boxplot_series(df, normalisation_method=None):
    """
    Treating each column as a separate boxplot and each row as an independent observation 
    (ie. different company)
    render a series of box plots to identify a shift in performance from the observations.
    normalisation_method should be one of the values present in 
    SectorSentimentSearchForm.normalisation_choices
    """
    # compute star performers: those who are above the mean on a given day counted over all days
    count = defaultdict(int)
    for col in df.columns:
        avg = df.mean(axis=0)
        winners = df[df[col] > avg[col]][col]
        for winner in winners.index:
            count[winner] += 1
    winner_results = []
    for asx_code, n_wins in count.items():
        x = df.loc[asx_code].sum()
        # avoid "dead cat bounce" stocks which fall spectacularly and then post major increases in percentage terms
        if x > 0.0:  
            winner_results.append((asx_code, n_wins, x))

    # and plot the normalised data
    if normalisation_method is None or normalisation_method == "1":
        normalized_df = df
        y_label = "Percentage change"
    elif normalisation_method == "2":
        normalized_df = (df - df.min()) / (df.max() - df.min())
        y_label = "Percentage change (min/max. scaled)"
    else:
        normalized_df = df / df.max(axis=0)  # div by max if all else fails...
        y_label = "Percentage change (normalised by dividing by max)"

    n_inches = len(df.columns) / 5
    melted = normalized_df.melt(ignore_index=False).dropna()
    plot = (
        p9.ggplot(melted, p9.aes(x="fetch_date", y="value"))
        + p9.geom_boxplot(outlier_colour="blue")
        + p9.theme(
            axis_text_x=p9.element_text(size=7),
            axis_text_y=p9.element_text(size=7),
            figure_size=(12, n_inches),
        )
        + p9.labs(x="Date (YYYY-MM-DD)", y=y_label)
        + p9.coord_flip()
    )
    return (
        plot,
        list(reversed(sorted(winner_results, key=lambda t: t[2]))),
    )

def plot_sector_field(df: pd.DataFrame, field, n_col=3):
    #print(df.columns)
    #assert set(df.columns) == set(['sector', 'date', 'mean_pe', 'sum_pe', 'sum_eps', 'mean_eps', 'n_stocks'])
    n_unique_sectors = df['sector'].nunique()
    df['date'] = pd.to_datetime(df['date'])
    plot = (
        p9.ggplot(df, p9.aes("date", field, group="sector", color="sector"))
        + p9.geom_line(size=1.0)
        + p9.facet_wrap("~sector", nrow=n_unique_sectors // n_col + 1, ncol=n_col, scales="free_y")
        + p9.xlab("")
        + p9.ylab(f"Sector-wide {field}")
        + p9.theme(
            axis_text_x=p9.element_text(angle=30, size=6),
            axis_text_y=p9.element_text(size=6),
            figure_size=(12, 6),
            panel_spacing=0.3,
            legend_position="none",
        )
    )

    return plot

