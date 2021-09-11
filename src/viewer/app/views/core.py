"""
Responsible for the core view - showing lists (paginated) of stocks handling all required corner conditions for missing data.
Other views are derived from this (both FBV and CBV) so be careful to take sub-views needs into account here.
"""
from django.db.models.query import QuerySet
from django.core.paginator import Paginator
from django.core.cache import cache
from django.shortcuts import render
from django.http import Http404, HttpResponse
from django.contrib.auth.decorators import login_required
from typing import Callable, Iterable
import plotnine as p9
import pandas as pd
import numpy as np
from lazydict import LazyDictionary
from app.models import (
    Timeframe,
    timing,
    stocks_by_sector,
    user_purchases,
    latest_quote,
    user_watchlist,
    selected_cached_stocks_cip,
    valid_quotes_only,
    all_available_dates,
    validate_date,
    validate_user,
    all_etfs,
)
from app.messages import info, warning, add_messages
from app.plots import plot_heatmap, plot_breakdown, user_theme
from app.data import cache_plot, timeframe_end_performance
from plotnine.layer import Layers
from plotnine.positions.position_dodge import position_dodge
from plotnine.themes.themeable import legend_position


@login_required
def png(request, key):
    """
    The cache contains all users of the site. In theory if key is guessed, then users can access each others content. For this reason
    all content is image only (not data) and the key is always a SHA256 hash with random content making it extremely difficult to guess.
    As further security PNG images served from the cache can only be accessed with a valid login.
    """
    assert (
        request is not None
    )  # dont yet require user login for this endpoint, but maybe we should???
    try:
        byte_value = cache.get(key)
        assert isinstance(byte_value, bytes)
        response = HttpResponse(byte_value, content_type="image/png")
        # response['X-Accel-Redirect'] = reader.name
        return response
    except KeyError:
        raise Http404("No such cached image: {}".format(key))


def positive_sum(col):
    return col[col > 0.0].sum()


def negative_sum(col):
    return col[col < 0.0].sum()


def plot_cumulative_returns(
    wanted_stocks: Iterable[str], ld: LazyDictionary
) -> p9.ggplot:
    df = ld["cip_df"]
    df = df.filter(wanted_stocks, axis=0).filter(regex="^\d", axis=1)
    dates = set(df.columns)
    movers = df
    movers["asx_code"] = movers.index
    movers = movers.melt(id_vars="asx_code", value_vars=dates)
    movers = movers[
        (movers["value"] < -5.0) | (movers["value"] > 5.0)
    ]  # ignore small movers
    # print(movers)
    movers["fetch_date"] = pd.to_datetime(movers["fetch_date"], format="%Y-%m-%d")

    # need to have separate dataframe's for positive and negative stocks - otherwise plotnine plot will be wrong
    pos_df = df.agg([positive_sum])
    neg_df = df.agg([negative_sum])
    pos_df = pos_df.melt(value_vars=dates)
    neg_df = neg_df.melt(value_vars=dates)
    pos_df["fetch_date"] = pd.to_datetime(pos_df["fetch_date"], format="%Y-%m-%d")
    neg_df["fetch_date"] = pd.to_datetime(neg_df["fetch_date"], format="%Y-%m-%d")

    plot = (
        p9.ggplot()
        + p9.geom_bar(
            p9.aes(x="fetch_date", y="value"),
            data=pos_df,
            stat="identity",
            fill="green",
        )
        + p9.geom_bar(
            p9.aes(x="fetch_date", y="value"),
            data=neg_df,
            stat="identity",
            fill="red",
        )
        + p9.geom_point(
            p9.aes(
                x="fetch_date",
                y="value",
                fill="asx_code",
            ),
            data=movers,
            size=3,
            position=p9.position_dodge(width=0.4),
            colour="black",
        )
    )
    return user_theme(
        plot,
        y_axis_label="Cumulative Return (%)",
        legend_position="right",
        asxtrade_want_cmap_d=False,
        asxtrade_want_fill_d=True,  # points (stocks) are filled with the user-chosen theme, but everything else is fixed
    )


def show_companies(
    matching_companies,  # may be QuerySet or iterable of stock codes (str)
    request,
    sentiment_timeframe: Timeframe,
    extra_context=None,
    template_name="all_stocks.html",
):
    """
    Support function to public-facing views to eliminate code redundancy
    """
    if isinstance(matching_companies, QuerySet):
        stocks_queryset = matching_companies  # we assume QuerySet is already sorted by desired criteria
    elif matching_companies is None or len(matching_companies) > 0:
        stocks_queryset, _ = latest_quote(matching_companies)
        # FALLTHRU
    else:
        # no companies to report?
        warning(request, "No matching companies.")
        return render(
            request, template_name, context={"timeframe": sentiment_timeframe}
        )

    # prune companies without a latest price, makes no sense to report them
    stocks_queryset = stocks_queryset.exclude(last_price__isnull=True)

    # sort queryset as this will often be requested by the USER
    arg = request.GET.get("sort_by", "asx_code")
    #info(request, "Sorting by {}".format(arg))

    if arg == "sector" or arg == "sector,-eps":
        ss = {
            s["asx_code"]: s["sector_name"]
            for s in stocks_by_sector().to_dict("records")
        }
        if arg == "sector":
            stocks_queryset = sorted(
                stocks_queryset, key=lambda s: ss.get(s.asx_code, "Z")
            )  # companies without sector sort last
        else:
            eps_dict = {
                s.asx_code: s.eps if s.eps is not None else 0.0 for s in stocks_queryset
            }
            stocks_queryset = sorted(
                stocks_queryset,
                key=lambda s: (ss.get(s.asx_code, "Z"), -eps_dict.get(s.asx_code, 0.0)),
            )
    else:
        sort_by = tuple(arg.split(","))
        stocks_queryset = stocks_queryset.order_by(*sort_by)

    # keep track of stock codes for template convenience
    asx_codes = [quote.asx_code for quote in stocks_queryset]
    n_top_bottom = (
        extra_context["n_top_bottom"] if "n_top_bottom" in extra_context else 20
    )
    print("show_companies: found {} stocks".format(len(asx_codes)))

    # setup context dict for the render
    context = {
        # NB: title and heatmap_title are expected to be supplied by caller via extra_context
        "timeframe": sentiment_timeframe,
        "title": "Caller must override",
        "watched": user_watchlist(request.user),
        "n_stocks": len(asx_codes),
        "n_top_bottom": n_top_bottom,
        "virtual_purchases": user_purchases(request.user),
    }

    # since we sort above, we must setup the pagination also...
    # assert isinstance(stocks_queryset, QuerySet)
    paginator = Paginator(stocks_queryset, 50)
    page_number = request.GET.get("page", 1)
    page_obj = paginator.page(page_number)
    context["page_obj"] = page_obj
    context["object_list"] = paginator

    # compute totals across all dates for the specified companies to look at top10/bottom10 in the timeframe
    ld = LazyDictionary()
    ld["cip_df"] = lambda ld: selected_cached_stocks_cip(asx_codes, sentiment_timeframe)
    ld["sum_by_company"] = lambda ld: ld["cip_df"].sum(axis=1, numeric_only=True)
    ld["top10"] = lambda ld: ld["sum_by_company"].nlargest(n_top_bottom)
    ld["bottom10"] = lambda ld: ld["sum_by_company"].nsmallest(n_top_bottom)
    ld["stocks_by_sector"] = lambda ld: stocks_by_sector()

    if len(asx_codes) <= 0 or len(ld["top10"]) <= 0:
        warning(request, "No matching companies found.")
    else:
        sorted_codes = "-".join(sorted(asx_codes))
        sentiment_heatmap_uri = cache_plot(
            f"{sorted_codes}-{sentiment_timeframe.description}-stocks-sentiment-plot",
            lambda ld: plot_heatmap(sentiment_timeframe, ld),
            datasets=ld,
        )

        key = f"{sorted_codes}-{sentiment_timeframe.description}-breakdown-plot"
        sector_breakdown_uri = cache_plot(key, plot_breakdown, datasets=ld)

        top10_plot_uri = cache_plot(
            f"top10-plot-{'-'.join(ld['top10'].index)}",
            lambda ld: plot_cumulative_returns(ld["top10"].index, ld),
            datasets=ld,
        )
        bottom10_plot_uri = cache_plot(
            f"bottom10-plot-{'-'.join(ld['bottom10'].index)}",
            lambda ld: plot_cumulative_returns(ld["bottom10"].index, ld),
            datasets=ld,
        )

        context.update(
            {
                "best_ten": ld["top10"],
                "worst_ten": ld["bottom10"],
                "sentiment_heatmap_uri": sentiment_heatmap_uri,
                "sentiment_heatmap_title": "{}: {}".format(
                    context["title"], sentiment_timeframe.description
                ),
                "sector_breakdown_uri": sector_breakdown_uri,
                "top10_plot_uri": top10_plot_uri,
                "bottom10_plot_uri": bottom10_plot_uri,
                "timeframe_end_performance": timeframe_end_performance(ld),
            }
        )

    if extra_context:
        context.update(extra_context)
    add_messages(request, context)
    # print(context)
    return render(request, template_name, context=context)


@login_required
def show_all_stocks(request):
    all_dates = all_available_dates()
    if len(all_dates) < 1:
        raise Http404("No ASX price data available!")
    ymd = all_dates[-1]
    validate_date(ymd)
    qs, _ = valid_quotes_only(ymd)
    timeframe = Timeframe()
    return show_companies(
        qs,
        request,
        timeframe,
        extra_context={
            "title": "All stocks",
            "sentiment_heatmap_title": "All stock sentiment: {}".format(
                timeframe.description
            ),
        },
    )


@login_required
def show_etfs(request):
    validate_user(request.user)
    matching_codes = all_etfs()
    extra_context = {
        "title": "Exchange Traded funds over past 300 days",
        "sentiment_heatmap_title": "Sentiment for ETFs",
    }
    return show_companies(
        matching_codes,
        request,
        Timeframe(),
        extra_context,
    )


# @login_required
# def show_increasing_eps_stocks(request):
#     validate_user(request.user)
#     timeframe = Timeframe(past_n_days=300)
#     sentiment_timeframe = Timeframe(past_n_days=30)
#     matching_companies = increasing_only_filter(None, timeframe, "eps")
#     return show_companies(
#         matching_companies,
#         request,
#         sentiment_timeframe,
#         extra_context={
#             "title": "Stocks with increasing EPS",
#             "sentiment_heatmap_title": "Sentiment for selected stocks",
#         },
#     )


# @login_required
# def show_increasing_yield_stocks(request):
#     validate_user(request.user)
#     timeframe = Timeframe(past_n_days=300)
#     sentiment_timeframe = Timeframe(past_n_days=30)
#     matching_companies = increasing_only_filter(
#         None, timeframe, "annual_dividend_yield", min_value=0.01
#     )
#     return show_companies(
#         matching_companies,
#         request,
#         sentiment_timeframe,
#         extra_context={
#             "title": "Stocks with increasing yield",
#             "sentiment_heatmap_title": "Sentiment for selected stocks",
#         },
#     )
