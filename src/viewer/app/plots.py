"""
Responsible for production of data visualisations and rendering this data as inline
base64 data for various django templates to use.
"""
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Iterable, Callable
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from pandas.core.base import NoNewAttributesMixin
import plotnine as p9
from lazydict import LazyDictionary
from django.contrib.auth import get_user_model
from app.models import (
    Timeframe,
    timing,
    user_purchases,
    all_available_dates,
)
from app.data import (
    make_portfolio_dataframe,
    cache_plot,
    make_portfolio_performance_dataframe,
    price_change_bins,
    calc_ma_crossover_points,
)
from plotnine.layer import Layers


def cached_portfolio_performance(user):
    assert isinstance(user, get_user_model())
    username = user.username
    overall_key = f"{username}-portfolio-performance"
    stock_key = f"{username}-stock-performance"
    contributors_key = f"{username}-contributor-performance"

    def data_factory(
        ld: LazyDictionary,
    ):  # dont create the dataframe unless we have to - avoid exxpensive call!
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
        timeframe = Timeframe(
            from_date=str(purchase_buy_dates[0]), to_date=all_available_dates()[-1]
        )

        return make_portfolio_performance_dataframe(stocks, timeframe, purchases)

    ld = LazyDictionary()
    ld["df"] = lambda ld: data_factory(ld)
    return (
        cache_plot(overall_key, plot_overall_portfolio, datasets=ld),
        cache_plot(stock_key, plot_portfolio_stock_performance, datasets=ld),
        cache_plot(contributors_key, plot_portfolio_contributors, datasets=ld),
    )


def user_theme(
    plot: p9.ggplot,
    x_axis_label: str = "",
    y_axis_label: str = "",
    title: str = "",
    **plot_theme,
) -> p9.ggplot:
    """Render the specified plot in the current theme with common attributes to all plots eg. legend_position etc. The themed plot is
    returned. Saves code in each plot by handled all the standard stuff here."""
    theme_args = {  # TODO FIXME... make defaults chosen from user profile
        "axis_text_x": p9.element_text(size=7),
        "axis_text_y": p9.element_text(size=7),
        "figure_size": (12, 6),
        "legend_position": "none",
    }
    theme_args.update(**plot_theme)

    # remove asxtrade kwargs
    want_cmap_d = theme_args.pop("asxtrade_want_cmap_d", True)
    want_fill_d = theme_args.pop(
        "asxtrade_want_fill_d", False
    )  # most graphs dont fill, so False by default
    want_fill_continuous = theme_args.pop("asxtrade_want_fill_continuous", False)
    plot = (
        plot
        + p9.theme_bw()  # TODO FIXME... make chosen theme from user profile
        + p9.labs(x=x_axis_label, y=y_axis_label, title=title)
        + p9.theme(**theme_args)
    )
    if want_cmap_d:
        plot += p9.scale_colour_cmap_d()
    if want_fill_d:
        plot += p9.scale_fill_cmap_d()
    elif want_fill_continuous:
        plot += p9.scale_fill_cmap()
    return plot


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
            if exclude_zero_bin and (
                bin_name == "0.0" or not isinstance(bin_name, str)
            ):
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
    bins, labels = price_change_bins()  # pylint: disable=unused-variable
    order = filter(
        lambda s: s != "0.0", labels
    )  # dont show the no change bin since it dominates the activity heatmap
    df["bin_ordered"] = pd.Categorical(df["bin"], categories=order)

    plot = p9.ggplot(df, p9.aes("date", "bin_ordered", fill="value")) + p9.geom_tile(
        show_legend=False
    )
    if plot_text_labels:
        plot = plot + p9.geom_text(p9.aes(label="value"), size=8, color="white")
    return user_theme(plot, y_axis_label="Daily change (%)")


@timing
def plot_fundamentals(
    df: pd.DataFrame,
    stock: str,
    line_size=1.5,  # pylint: disable=unused-argument
    columns_to_report=(
        "pe",
        "eps",
        "annual_dividend_yield",
        "volume",
        "last_price",
        "change_in_percent_cumulative",
        "change_price",
        "market_cap",
        "number_of_shares",
    ),
) -> str:
    plot_df = pd.melt(
        df,
        id_vars="fetch_date",
        value_vars=columns_to_report,
        var_name="indicator",
        value_name="value",
    )
    plot_df = plot_df[plot_df["indicator"].isin(columns_to_report)]
    plot_df["value"] = pd.to_numeric(plot_df["value"])
    plot_df = plot_df.dropna(axis=0, subset=["value"], how="any")
    n = len(columns_to_report)
    plot = (
        p9.ggplot(
            plot_df,
            p9.aes("fetch_date", "value", colour="indicator"),
        )
        + p9.geom_path(show_legend=False, size=line_size)
        + p9.facet_wrap("~ indicator", nrow=n, ncol=1, scales="free_y")
    )
    return user_theme(plot, figure_size=(12, n))


def plot_overall_portfolio(
    ld: LazyDictionary,
    figure_size=(12, 4),
    line_size=1.5,
    date_text_size=7,
) -> p9.ggplot:
    """
    Given a daily snapshot of virtual purchases plot both overall and per-stock
    performance. Return a ggplot instance representing the visualisation
    """
    portfolio_df = ld["df"]
    df = portfolio_df.filter(
        items=["portfolio_cost", "portfolio_worth", "portfolio_profit", "date"]
    )
    df = df.melt(id_vars=["date"], var_name="field")
    plot = (
        p9.ggplot(df, p9.aes("date", "value", group="field", color="field"))
        + p9.geom_line(size=line_size)
        + p9.facet_wrap("~ field", nrow=3, ncol=1, scales="free_y")
    )
    return user_theme(
        plot,
        y_axis_label="$ AUD",
        axis_text_x=p9.element_text(angle=30, size=date_text_size),
    )


def plot_portfolio_contributors(ld: LazyDictionary, figure_size=(11, 5)) -> p9.ggplot:
    df = ld["df"]
    melted_df = make_portfolio_dataframe(df, melt=True)

    all_dates = sorted(melted_df["date"].unique())
    df = melted_df[melted_df["date"] == all_dates[-1]]
    # print(df)
    df = df[df["field"] == "stock_profit"]  # only latest profit is plotted
    df["contribution"] = [
        "positive" if profit >= 0.0 else "negative" for profit in df["value"]
    ]

    # 2. plot contributors ie. winners and losers
    plot = (
        p9.ggplot(df, p9.aes("stock", "value", group="stock", fill="stock"))
        + p9.geom_bar(stat="identity")
        + p9.facet_grid("contribution ~ field", scales="free_y")
    )
    return user_theme(
        plot, y_axis_label="$ AUD", figure_size=figure_size, asxtrade_want_fill_d=True
    )


def plot_portfolio_stock_performance(
    ld: LazyDictionary, figure_width: int = 12, date_text_size=7
) -> p9.ggplot:

    df = ld["df"]
    df = df[df["stock_cost"] > 0.0]

    # latest_date = df.iloc[-1, 6]
    # latest_profit = df[df["date"] == latest_date]
    # print(df)
    pivoted_df = df.pivot(index="stock", columns="date", values="stock_profit")
    latest_date = pivoted_df.columns[-1]
    # print(latest_date)
    mean_profit = pivoted_df.mean(axis=1)
    n_stocks = len(mean_profit)
    # if we want ~4 stocks per facet plot, then we need to specify the appropriate calculation for df.qcut()
    bins = pd.qcut(mean_profit, int(100 / n_stocks) + 1)
    # print(bins)
    df = df.merge(bins.to_frame(name="bins"), left_on="stock", right_index=True)
    # print(df)
    textual_df = df[df["date"] == latest_date]
    # print(textual_df)
    # melted_df = make_portfolio_dataframe(df, melt=True)

    plot = (
        p9.ggplot(df, p9.aes("date", "stock_profit", group="stock", colour="stock"))
        + p9.geom_smooth(size=1.0, span=0.3, se=False)
        + p9.facet_wrap("~bins", ncol=1, nrow=len(bins), scales="free_y")
        + p9.geom_text(
            p9.aes(x="date", y="stock_profit", label="stock"),
            color="black",
            size=9,
            data=textual_df,
            position=p9.position_jitter(width=10, height=10),
        )
    )
    return user_theme(
        plot,
        y_axis_label="$ AUD",
        figure_size=(figure_width, int(len(bins) * 1.2)),
        axis_text_x=p9.element_text(angle=30, size=date_text_size),
    )


def plot_company_rank(ld: LazyDictionary) -> p9.ggplot:
    df = ld["rank"]
    # assert 'sector' in df.columns
    n_bin = len(df["bin"].unique())
    # print(df)
    plot = (
        p9.ggplot(df, p9.aes("date", "rank", group="asx_code", color="asx_code"))
        + p9.geom_smooth(span=0.3, se=False)
        + p9.geom_text(
            p9.aes(label="asx_code", x="x", y="y"),
            nudge_x=1.2,
            size=6,
            show_legend=False,
        )
        + p9.facet_wrap("~bin", nrow=n_bin, ncol=1, scales="free_y")
    )
    return user_theme(
        plot,
        figure_size=(12, 20),
        subplots_adjust={"right": 0.8},
    )


def plot_company_versus_sector(
    df: pd.DataFrame, stock: str, sector: str  # pylint: disable=unused-argument
) -> p9.ggplot:
    if df is None or len(df) < 1:
        print("No data for stock vs. sector plot... ignored")
        return None

    df["date"] = pd.to_datetime(df["date"])
    # print(df)
    plot = p9.ggplot(
        df, p9.aes("date", "value", group="group", color="group", fill="group")
    ) + p9.geom_line(size=1.5)
    return user_theme(
        plot,
        y_axis_label="Change since start (%)",
        subplots_adjust={"right": 0.8},
        legend_position="right",
    )


def plot_market_wide_sector_performance(ld: LazyDictionary) -> p9.ggplot:
    """
    Display specified dates for average sector performance. Each company is assumed to have at zero
    at the start of the observation period. A plot as base64 data is returned.
    """
    all_stocks_cip = ld["sector_cumsum_df"]
    n_stocks = len(all_stocks_cip)
    # merge in sector information for each company
    code_and_sector = ld["stocks_by_sector"]
    n_unique_sectors = len(code_and_sector["sector_name"].unique())
    print("Found {} unique sectors".format(n_unique_sectors))

    # print(df)
    # print(code_and_sector)
    df = all_stocks_cip.merge(code_and_sector, left_index=True, right_on="asx_code")
    print(
        "Found {} stocks, {} sectors and merged total: {}".format(
            n_stocks, len(code_and_sector), len(df)
        )
    )
    # print(df)
    grouped_df = df.groupby("sector_name").mean()
    # print(grouped_df)

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
    )
    return user_theme(
        plot,
        y_axis_label="Average sector change (%)",
        panel_spacing=0.3,
        axis_text_x=p9.element_text(angle=30, size=7),
    )


def plot_series(
    df,
    x=None,
    y=None,
    tick_text_size=6,
    line_size=1.5,
    y_axis_label="Point score",
    x_axis_label="",
    color="stock",
    use_smooth_line=False,
):
    if df is None or len(df) < 1:
        return None

    assert len(x) > 0 and len(y) > 0
    assert line_size > 0.0
    assert isinstance(tick_text_size, int) and tick_text_size > 0
    assert y_axis_label is not None
    assert x_axis_label is not None
    args = {"x": x, "y": y}
    if color:
        args["color"] = color
    plot = p9.ggplot(df, p9.aes(**args))
    if use_smooth_line:
        plot += p9.geom_smooth(
            size=line_size, span=0.3, se=False
        )  # plotnine doesnt support confidence intervals with Loess smoothings, so se=False
    else:
        plot += p9.geom_line(size=line_size)
    return user_theme(
        plot,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        axis_text_x=p9.element_text(angle=30, size=tick_text_size),
        axis_text_y=p9.element_text(size=tick_text_size),
    )


def plot_market_cap_distribution(ld: LazyDictionary) -> p9.ggplot:
    df = ld["market_cap_df"]
    assert set(df.columns).intersection(set(["market", "market_cap", "bin"])) == set(
        ["market", "market_cap", "bin"]
    )
    pos_market_cap_only = df[df["market_cap"] > 0.0]
    plot = (
        p9.ggplot(pos_market_cap_only)
        + p9.geom_boxplot(p9.aes(x="market", y="market_cap"))
        + p9.facet_wrap("bin", scales="free_y")
        + p9.scales.scale_y_log10()
    )
    return user_theme(
        plot,
        y_axis_label="Market cap. ($AUD Millions)",
        subplots_adjust={"wspace": 0.30},
    )


def plot_breakdown(ld: LazyDictionary) -> p9.ggplot:
    """Stacked bar plot of increasing and decreasing stocks per sector in the specified df"""
    cip_df = ld["cip_df"]

    cols_to_drop = [colname for colname in cip_df.columns if colname.startswith("bin_")]
    df = cip_df.drop(columns=cols_to_drop)
    df = pd.DataFrame(df.sum(axis="columns"), columns=["sum"])
    ss = ld["stocks_by_sector"]
    # ss should be:
    #             asx_code             sector_name
    # asx_code
    # 14D           14D             Industrials
    # 1AD           1AD             Health Care
    # 1AG           1AG             Industrials
    # 1AL           1AL  Consumer Discretionary........
    # print(ss)
    df = df.merge(ss, left_index=True, right_index=True)

    if len(df) == 0:  # no stock in cip_df have a sector? ie. ETF?
        return None

    assert set(df.columns) == set(["sum", "asx_code", "sector_name"])
    df["increasing"] = df.apply(
        lambda row: "up" if row["sum"] >= 0.0 else "down", axis=1
    )
    sector_names = (
        df["sector_name"].value_counts().index.tolist()
    )  # sort bars by value count (ascending)
    sector_names_cat = pd.Categorical(df["sector_name"], categories=sector_names)
    df = df.assign(sector_name_cat=sector_names_cat)

    # print(df)
    plot = (
        p9.ggplot(df, p9.aes(x="factor(sector_name_cat)", fill="factor(increasing)"))
        + p9.geom_bar()
        + p9.coord_flip()
    )
    return user_theme(
        plot,
        x_axis_label="Sector",
        y_axis_label="Number of stocks",
        subplots_adjust={"left": 0.2, "right": 0.85},
        legend_title=p9.element_blank(),
        asxtrade_want_fill_d=True,
    )


def plot_heatmap(
    timeframe: Timeframe, ld: LazyDictionary, bin_cb=price_change_bins
) -> p9.ggplot:
    """
    Plot the specified data matrix as binned values (heatmap) with X axis being dates over the specified timeframe and Y axis being
    the percentage change on the specified date (other metrics may also be used, but you will likely need to adjust the bins)
    :rtype: p9.ggplot instance representing the heatmap
    """
    df = ld["cip_df"]
    bins, labels = bin_cb()
    # print(df.columns)
    # print(bins)
    try:
        # NB: this may fail if no prices are available so we catch that error and handle accordingly...
        for date in df.columns:
            df["bin_{}".format(date)] = pd.cut(df[date], bins, labels=labels)
        sentiment_plot = make_sentiment_plot(
            df, plot_text_labels=timeframe.n_days <= 30
        )  # show counts per bin iff not too many bins
        return sentiment_plot
    except KeyError:
        return None


def plot_sector_performance(dataframe: pd.DataFrame, descriptor: str):
    assert len(dataframe) > 0
    dataframe["date"] = pd.to_datetime(dataframe["date"], format="%Y-%m-%d")

    # now do the plot
    labels = [
        "Number of stocks up >5%",
        "Number of stocks down >5%",
        "Remaining stocks",
    ]
    # print(dataframe)
    dataframe.columns = labels + ["date"]
    melted_df = dataframe.melt(value_vars=labels, id_vars="date")
    plot = (
        p9.ggplot(
            melted_df,
            p9.aes("date", "value", colour="variable", group="factor(variable)"),
        )
        + p9.facet_wrap("~variable", ncol=1, scales="free_y")
        + p9.geom_line(size=1.3)
    )
    return user_theme(plot)


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


@timing
def plot_momentum(stock: str, timeframe: Timeframe, ld: LazyDictionary) -> plt.Figure:
    assert len(stock) > 0
    assert "stock_df" in ld or "stock_df_200" in ld
    start_date = timeframe.earliest_date
    stock_df = ld["stock_df_200"] if "stock_df_200" in ld else ld["stock_df"]
    last_price = stock_df["last_price"]
    volume = stock_df["volume"]
    day_low_price = stock_df["day_low_price"]
    day_high_price = stock_df["day_high_price"]
    # print(last_price)
    # print(volume)
    # print(day_low_price)
    # print(day_high_price)

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

    timeline = pd.to_datetime(last_price.index, format="%Y-%m-%d")
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
    ma200 = last_price.rolling(window=200, min_periods=50).mean()

    # timeline = timeline.to_list()
    (linema20,) = ax2.plot(timeline, ma20, color="blue", lw=2, label="MA (20)")
    (linema200,) = ax2.plot(timeline, ma200, color="red", lw=2, label="MA (200)")
    assert linema20 is not None
    assert linema200 is not None

    props = font_manager.FontProperties(size=10)
    leg = ax2.legend(loc="lower left", shadow=True, fancybox=True, prop=props)
    leg.get_frame().set_alpha(0.5)

    volume = (last_price * volume) / 1e6  # dollar volume in millions
    # print(volume)
    vmax = np.nanmax(volume)
    # print(vmax)
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
        plt.xlim(left=datetime.strptime(start_date, "%Y-%m-%d"))
    except IndexError:
        print("WARNING: unable to set plot start_date - things may look weird")
    plt.plot()
    fig = plt.gcf()
    plt.close(fig)
    return fig


@timing
def plot_trend(sample_period="M", ld: LazyDictionary = None) -> str:
    """
    Given a dataframe of a single stock from company_prices() this plots the highest price
    in each month over the time period of the dataframe.
    """
    assert "stock_df" in ld

    def inner_date_fmt(dates_to_format):
        results = []
        for d in dates_to_format:
            d -= timedelta(
                weeks=4
            )  # breaks are set to the end of the month rather than the start... so
            results.append(d.strftime("%Y-%m"))
        return results

    stock_df = ld["stock_df"]
    # print(stock_df)
    dataframe = stock_df.filter(items=["last_price"])
    dataframe.index = pd.to_datetime(dataframe.index, format="%Y-%m-%d")
    dataframe = dataframe.resample(sample_period).max()
    # print(dataframe)
    plot = (
        p9.ggplot(
            dataframe,
            p9.aes(
                x="dataframe.index", y=dataframe.columns[0], fill=dataframe.columns[0]
            ),
        )
        + p9.geom_bar(stat="identity", alpha=0.7)
        + p9.scale_x_datetime(
            labels=inner_date_fmt
        )  # dont print day (always 1st day of month due to resampling)
    )
    return user_theme(plot, y_axis_label="$ AUD", asxtrade_want_fill_continuous=True)


def plot_points_by_rule(net_points_by_rule: defaultdict(int)) -> p9.ggplot:
    if net_points_by_rule is None or len(net_points_by_rule) < 1:
        return None

    rows = []
    for k, v in net_points_by_rule.items():
        rows.append({"rule": str(k), "net_points": v})
    df = pd.DataFrame.from_records(rows)
    plot = (
        p9.ggplot(df, p9.aes(x="rule", y="net_points", fill="net_points"))
        + p9.geom_bar(stat="identity", alpha=0.7)
        + p9.coord_flip()
    )
    return user_theme(
        plot,
        x_axis_label="Rule",
        y_axis_label="Contributions to points by rule",
        subplots_adjust={"left": 0.2},
        asxtrade_want_fill_continuous=True,
    )


def plot_boxplot_series(df, normalisation_method=None):
    """
    Treating each column as a separate boxplot and each row as an independent observation
    (ie. different company)
    render a series of box plots to identify a shift in performance from the observations.
    normalisation_method should be one of the values present in
    SectorSentimentSearchForm.normalisation_choices
    """

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
        + p9.coord_flip()
    )
    return user_theme(plot, y_axis_label=y_label, figure_size=(12, n_inches))


def plot_sector_field(df: pd.DataFrame, field, n_col=3):
    # print(df.columns)
    # assert set(df.columns) == set(['sector', 'date', 'mean_pe', 'sum_pe', 'sum_eps', 'mean_eps', 'n_stocks'])
    n_unique_sectors = df["sector"].nunique()
    df["date"] = pd.to_datetime(df["date"])
    plot = (
        p9.ggplot(df, p9.aes("date", field, group="sector", color="sector"))
        + p9.geom_line(size=1.0)
        + p9.facet_wrap(
            "~sector", nrow=n_unique_sectors // n_col + 1, ncol=n_col, scales="free_y"
        )
    )

    return user_theme(
        plot,
        y_axis_label=f"Sector-wide {field}",
        panel_spacing=0.3,
        axis_text_x=p9.element_text(angle=30, size=7),
    )


def plot_sector_top_eps_contributors(
    df: pd.DataFrame, stocks_by_sector_df: pd.DataFrame
) -> p9.ggplot:
    """
    Returns a plot of the top 20 contributors per sector, based on the most recent EPS value per stock in the dataframe. If no
    stocks in a given sector have positive EPS, the sector will not be plotted.
    """
    most_recent_date = df.columns[-1]
    last_known_eps = df[most_recent_date]
    last_known_eps = last_known_eps[last_known_eps >= 0.0].to_frame()
    # print(stocks_by_sector_df)
    last_known_eps = last_known_eps.merge(
        stocks_by_sector_df, left_index=True, right_on="asx_code"
    )
    last_known_eps["rank"] = last_known_eps.groupby("sector_name")[
        most_recent_date
    ].rank("dense", ascending=False)
    last_known_eps = last_known_eps[last_known_eps["rank"] <= 10.0]
    n_sectors = last_known_eps["sector_name"].nunique()
    last_known_eps["eps"] = last_known_eps[most_recent_date]

    plot = (
        p9.ggplot(
            last_known_eps,
            p9.aes(
                y="eps",
                x="reorder(asx_code,eps)",  # sort bars by eps within each sub-plot
                group="sector_name",
                fill="sector_name",
            ),
        )
        + p9.geom_bar(stat="identity")
        + p9.facet_wrap("~sector_name", ncol=1, nrow=n_sectors, scales="free")
        + p9.coord_flip()
    )
    return user_theme(
        plot,
        y_axis_label="EPS ($AUD)",
        x_axis_label="Top 10 ASX stocks per sector as at {}".format(most_recent_date),
        subplots_adjust={"hspace": 0.4},
        figure_size=(12, int(n_sectors * 1.5)),
        asxtrade_want_cmap_d=False,
        asxtrade_want_fill_d=True,
    )


def plot_monthly_returns(
    timeframe: Timeframe, stock: str, ld: LazyDictionary
) -> p9.ggplot:
    start = timeframe.earliest_date
    end = timeframe.most_recent_date
    dt = pd.date_range(start, end, freq="BMS")
    df = ld["stock_df"]
    # print(df)
    df = df.filter([d.strftime("%Y-%m-%d") for d in dt], axis=0)
    df["percentage_change"] = df["last_price"].pct_change(periods=1) * 100.0
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
    df = df.fillna(0.0)  # NB: avoid plotnine warning plotting missing data
    # print(df)

    plot = p9.ggplot(
        df, p9.aes(x="df.index", y="percentage_change", fill="percentage_change")
    ) + p9.geom_bar(stat="identity")
    return user_theme(
        plot, asxtrade_want_cmap_d=False, asxtrade_want_fill_continuous=True
    )


def plot_sector_monthly_mean_returns(ld: LazyDictionary) -> dict:
    all_stocks = ld["monthly_returns_by_stock"]
    ret = {}
    ss = ld["stocks_by_sector"]
    all_stock_average_df = all_stocks.mean(axis=1).to_frame(name="average")
    all_stock_average_df["dataset"] = "All stock average"
    final_df = all_stock_average_df
    # print(ss)
    for current_sector in ss["sector_name"].unique():
        # print(current_sector)
        wanted_stocks = set(ss[ss["sector_name"] == current_sector]["asx_code"])
        # print(wanted_stocks)
        df = (
            all_stocks.filter(items=wanted_stocks, axis="columns")
            .mean(axis=1)
            .to_frame(name="average")
        )
        df["dataset"] = current_sector
        final_df = final_df.append(df)

    final_df["date"] = pd.to_datetime(final_df.index, format="%Y-%m-%d")
    plot = (
        p9.ggplot(final_df, p9.aes(x="date", y="average"))
        + p9.geom_bar(stat="identity")
        + p9.facet_wrap("~dataset", ncol=2, scales="free_y")
    )
    ret["month-by-month-average-returns"] = cache_plot(
        "monthly-mean-returns",
        lambda ld: user_theme(
            plot,
            y_axis_label="Average percent return per month",
            figure_size=(12, 10),
            subplots_adjust={"wspace": 0.15},
            axis_text_x=p9.element_text(angle=30, size=7),
        ),
    )
    return ret
