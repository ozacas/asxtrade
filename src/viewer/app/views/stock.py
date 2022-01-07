"""
Responsible for providing detiled views about a single stock and closely related views
"""
from typing import Iterable
from matplotlib.pyplot import subplots_adjust
import pandas as pd
import plotnine as p9
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import Http404
from lazydict import LazyDictionary
from app.models import (
    latest_quote,
    Quotation,
    stocks_by_sector,
    valid_quotes_only,
    validate_stock,
    validate_user,
    stock_info,
    cached_all_stocks_cip,
    companies_with_same_sector,
    Timeframe,
    company_prices,
    user_watchlist,
    selected_cached_stocks_cip,
    timing,
    financial_metrics,
    all_stock_fundamental_fields
)
from app.analysis import (
    default_point_score_rules,
    rank_cumulative_change,
    calculate_trends,
)
from app.messages import warning
from app.data import (
    cache_plot,
    make_point_score_dataframe,
    label_shorten,
    pe_trends_df,
    make_stock_vs_sector_dataframe,
)
from app.plots import (
    user_theme,
    plot_points_by_rule,
    plot_fundamentals,
    plot_momentum,
    plot_trend,
    plot_series,
    plot_monthly_returns,
    plot_company_rank,
    cached_portfolio_performance,
    plot_company_versus_sector,
)
from plotnine.guides.guide_colorbar import guide_colorbar


@timing
def fundamentals_dataframe(
    timeframe: Timeframe, stock: str, ld: LazyDictionary
) -> pd.DataFrame:
    """Return a dict of the fundamentals plots for the current django template render to use"""
    df = ld["stock_df"]
    # print(df)
    df["change_in_percent_cumulative"] = df[
        "change_in_percent"
    ].cumsum()  # nicer to display cumulative
    df = df.drop("change_in_percent", axis=1)
    df["volume"] = df["last_price"] * df["volume"] / 1000000  # again, express as $(M)
    df["market_cap"] /= 1000 * 1000
    df["number_of_shares"] /= 1000 * 1000
    df["fetch_date"] = pd.to_datetime(df.index, format="%Y-%m-%d")
    # print(df.shape)
    df = df.set_index("fetch_date")
    df = df.resample(
        "B"
    ).asfreq()  # fill gaps in dataframe with business day dates only
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(df)
    df["fetch_date"] = pd.to_datetime(df.index, format="%Y-%m-%d")
    return df


@login_required
def show_financial_metrics(request, stock=None):
    validate_user(request.user)
    validate_stock(stock)

    def data_factory(ld: LazyDictionary):
        data_df = financial_metrics(stock)
        if data_df is None or len(data_df) < 1:
            raise Http404(f"No financial metrics available for {stock}")
        return data_df

    def find_linear_metrics(ld: LazyDictionary) -> Iterable[str]:
        linear_metrics = calculate_trends(ld["data_df"])
        good_linear_metrics = []
        for k, t in linear_metrics.items():
            if t[1] < 0.1:
                good_linear_metrics.append(k)
        return good_linear_metrics

    def find_exp_metrics(ld: LazyDictionary) -> Iterable[str]:
        exp_metrics = calculate_trends(
            ld["data_df"], polynomial_degree=2, nrmse_cutoff=0.05
        )
        good_linear_metrics = set(ld["linear_metrics"])
        good_exp_metrics = []
        for k, t in exp_metrics.items():
            if t[1] < 0.1 and k not in good_linear_metrics:
                good_exp_metrics.append(k)
        return good_exp_metrics

    ld = LazyDictionary()
    ld["data_df"] = lambda ld: data_factory(ld)
    ld["linear_metrics"] = lambda ld: find_linear_metrics(ld)
    ld["exp_metrics"] = lambda ld: find_exp_metrics(ld)

    # print(
    #     f"n_metrics == {len(data_df)} n_trending={len(linear_metrics.keys())} n_good_fit={len(good_linear_metrics)} n_good_exp={len(good_exp_metrics)}"
    # )

    def plot_metrics(df: pd.DataFrame, use_short_labels=False, **kwargs):
        plot = (
            p9.ggplot(df, p9.aes(x="date", y="value", colour="metric"))
            + p9.geom_line(size=1.3)
            + p9.geom_point(size=3)
        )
        if use_short_labels:
            plot += p9.scale_y_continuous(labels=label_shorten)
        n_metrics = df["metric"].nunique()
        return user_theme(
            plot,
            subplots_adjust={"left": 0.2},
            figure_size=(12, int(n_metrics * 1.5)),
            **kwargs,
        )

    def plot_linear_trending_metrics(ld: LazyDictionary):
        df = ld["data_df"].filter(ld["linear_metrics"], axis=0)
        if len(df) < 1:
            return None
        df["metric"] = df.index
        df = df.melt(id_vars="metric").dropna(how="any", axis=0)
        plot = plot_metrics(df, use_short_labels=True)
        plot += p9.facet_wrap("~metric", ncol=1, scales="free_y")
        return plot

    def plot_exponential_growth_metrics(ld: LazyDictionary):
        df = ld["data_df"].filter(ld["exp_metrics"], axis=0)
        if len(df) < 1:
            return None
        df["metric"] = df.index
        df = df.melt(id_vars="metric").dropna(how="any", axis=0)
        plot = plot_metrics(df)
        plot += p9.facet_wrap("~metric", ncol=1, scales="free_y")

        return plot

    def plot_earnings_and_revenue(ld: LazyDictionary):
        df = ld["data_df"].filter(["Ebit", "Total Revenue", "Earnings"], axis=0)
        if len(df) < 2:
            print(f"WARNING: revenue and earnings not availabe for {stock}")
            return None
        df["metric"] = df.index
        df = df.melt(id_vars="metric").dropna(how="any", axis=0)
        plot = plot_metrics(
            df,
            use_short_labels=True,
            legend_position="right",
            y_axis_label="$ AUD",
        )  # need to show metric name somewhere on plot
        return plot

    er_uri = cache_plot(
        f"{stock}-earnings-revenue-plot",
        lambda ld: plot_earnings_and_revenue(ld),
        datasets=ld,
    )
    trending_metrics_uri = cache_plot(
        f"{stock}-trending-metrics-plot",
        lambda ld: plot_linear_trending_metrics(ld),
        datasets=ld,
    )
    exp_growth_metrics_uri = cache_plot(
        f"{stock}-exponential-growth-metrics-plot",
        lambda ld: plot_exponential_growth_metrics(ld),
        datasets=ld,
    )
    warning(
        request,
        "Due to experimental data ingest - data on this page may be wrong/misleading/inaccurate/missing. Use at own risk.",
    )
    context = {
        "asx_code": stock,
        "data": ld["data_df"],
        "earnings_and_revenue_plot_uri": er_uri,
        "trending_metrics_plot_uri": trending_metrics_uri,
        "exp_growth_metrics_plot_uri": exp_growth_metrics_uri,
    }
    return render(request, "stock_financial_metrics.html", context=context)


@login_required
def show_stock(request, stock=None, n_days=2 * 365):
    """
    Displays a view of a single stock via the template and associated state
    """
    validate_stock(stock)
    validate_user(request.user)
    plot_timeframe = Timeframe(past_n_days=n_days)  # for template

    def dataframe(ld: LazyDictionary) -> pd.DataFrame:
        momentum_timeframe = Timeframe(
            past_n_days=n_days + 200
        )  # to warmup MA200 function
        df = company_prices(
            (stock,),
            momentum_timeframe,
            fields=all_stock_fundamental_fields,
            missing_cb=None,
        )
        return df

    # key dynamic images and text for HTML response. We only compute the required data if image(s) not cached
    # print(df)
    ld = LazyDictionary()
    ld["stock_df"] = lambda ld: ld["stock_df_200"].filter(
        items=plot_timeframe.all_dates(), axis="rows"
    )
    ld["cip_df"] = lambda: cached_all_stocks_cip(plot_timeframe)
    ld["stock_df_200"] = lambda ld: dataframe(ld)
    ld["sector_companies"] = lambda: companies_with_same_sector(stock)
    ld["company_details"] = lambda: stock_info(stock, lambda msg: warning(request, msg))
    ld["sector"] = lambda ld: ld["company_details"].get("sector_name", "")
    # point_score_results is a tuple (point_score_df, net_points_by_rule)
    ld["point_score_results"] = lambda ld: make_point_score_dataframe(
        stock, default_point_score_rules(), ld
    )
    ld["stock_vs_sector_df"] = lambda ld: make_stock_vs_sector_dataframe(
        ld["cip_df"], stock, ld["sector_companies"]
    )
    print(ld["stock_vs_sector_df"])

    momentum_plot = cache_plot(
        f"{plot_timeframe.description}-{stock}-rsi-plot",
        lambda ld: plot_momentum(stock, plot_timeframe, ld),
        datasets=ld,
    )
    monthly_maximum_plot = cache_plot(
        f"{plot_timeframe.description}-{stock}-monthly-maximum-plot",
        lambda ld: plot_trend("M", ld),
        datasets=ld,
    )
    monthly_returns_plot = cache_plot(
        f"{plot_timeframe.description}-{stock}-monthly returns",
        lambda ld: plot_monthly_returns(plot_timeframe, stock, ld),
        datasets=ld,
    )
    company_versus_sector_plot = cache_plot(
        f"{stock}-{ld['sector']}-company-versus-sector",
        lambda ld: plot_company_versus_sector(
            ld["stock_vs_sector_df"], stock, ld["sector"]
        ),
        datasets=ld,
    )

    point_score_plot = cache_plot(
        f"{plot_timeframe.description}-{stock}-point-score-plot",
        lambda ld: plot_series(ld["point_score_results"][0], x="date", y="points"),
        datasets=ld,
    )
    net_rule_contributors_plot = cache_plot(
        f"{plot_timeframe.description}-{stock}-rules-by-points",
        lambda ld: plot_points_by_rule(ld["point_score_results"][1]),
        datasets=ld,
    )

    # populate template and render HTML page with context
    context = {
        "asx_code": stock,
        "watched": user_watchlist(request.user),
        "timeframe": plot_timeframe,
        "information": ld["company_details"],
        "momentum": {
            "rsi_plot": momentum_plot,
            "monthly_highest_price": {
                "title": "Highest price each month",
                "plot_uri": monthly_maximum_plot,
            },
        },
        "fundamentals": {
            "plot_uri": cache_plot(
                f"{stock}-{plot_timeframe.description}-fundamentals-plot",
                lambda ld: plot_fundamentals(
                    fundamentals_dataframe(plot_timeframe, stock, ld),
                    stock,
                ),
                datasets=ld,
            ),
            "title": "Stock fundamentals: EPS, PE, DY etc.",
            "timeframe": plot_timeframe,
        },
        "stock_vs_sector": {
            "plot_uri": company_versus_sector_plot,
            "title": "Company versus sector - percentage change",
            "timeframe": plot_timeframe,
        },
        "point_score": {
            "plot_uri": point_score_plot,
            "title": "Points score due to price movements",
        },
        "net_contributors": {
            "plot_uri": net_rule_contributors_plot,
            "title": "Contributions to point score by rule",
        },
        "month_by_month_return_uri": monthly_returns_plot,
    }
    return render(request, "stock_page.html", context=context)


@login_required
def show_trends(request):
    user = request.user
    validate_user(user)
    stocks = user_watchlist(user)
    timeframe = Timeframe(past_n_days=300)
    ld = LazyDictionary()
    ld["cip_df"] = lambda ld: selected_cached_stocks_cip(stocks, timeframe)
    ld["trends"] = lambda ld: calculate_trends(ld["cip_df"])
    ld["rank"] = lambda ld: rank_cumulative_change(
        ld["cip_df"].filter(ld["trends"].keys(), axis="index"), timeframe
    )

    trending_companies_plot = cache_plot(
        f"{user.username}-watchlist-trends",
        lambda ld: plot_company_rank(ld),
        datasets=ld,
    )

    context = {
        "watchlist_trends": ld["trends"],
        "timeframe": timeframe,
        "trending_companies_uri": trending_companies_plot,
        "trending_companies_plot_title": "Trending watchlist stocks (ranked): {}".format(
            timeframe.description
        ),
    }
    return render(request, "watchlist-rank.html", context=context)


@login_required
def show_purchase_performance(request):
    validate_user(request.user)

    (
        portfolio_performance_uri,
        stock_performance_uri,
        contributors_uri,
    ) = cached_portfolio_performance(request.user)

    context = {
        "title": "Portfolio performance",
        "portfolio_title": "Overall",
        "performance_uri": portfolio_performance_uri,
        "stock_title": "Stock",
        "stock_performance_uri": stock_performance_uri,
        "contributors_uri": contributors_uri,
    }
    return render(request, "portfolio_trends.html", context=context)


@login_required
def show_total_earnings(request):
    validate_user(request.user)

    def data_factory(df: pd.DataFrame) -> pd.DataFrame:
        df = df.pivot(
            index=["asx_code", "fetch_date"], columns="field_name", values="field_value"
        )
        required = (df.number_of_shares > 0) & (df.eps > 0.0)
        df = df[required]  # ignore stocks which have unknowns
        # print(df)
        df["total_earnings"] = df["eps"] * df["number_of_shares"]
        df = df.dropna(how="any", axis=0)
        df = df.reset_index()
        df = df.pivot(index="asx_code", columns="fetch_date", values="total_earnings")
        df = df.merge(stocks_by_sector(), left_index=True, right_on="asx_code")
        df = df.set_index("asx_code", drop=True)
        df = df.groupby("sector_name").sum()
        df["sector_name"] = df.index
        df = df.melt(id_vars="sector_name", var_name="fetch_date")
        assert set(df.columns) == set(["sector_name", "fetch_date", "value"])
        df["fetch_date"] = pd.to_datetime(df["fetch_date"], format="%Y-%m-%d")

        return df

    def plot(df: pd.DataFrame) -> p9.ggplot:
        plot = (
            p9.ggplot(
                df,
                p9.aes(
                    x="fetch_date",
                    y="value",
                    color="sector_name",  # group="sector_name"
                ),
            )
            + p9.geom_line(size=1.2)
            + p9.facet_wrap("~sector_name", ncol=2, scales="free_y")
            + p9.scale_y_continuous(labels=label_shorten)
        )
        return user_theme(
            plot,
            y_axis_label="Total sector earnings ($AUD, positive contributions only)",
            figure_size=(12, 14),
            subplots_adjust={"wspace": 0.25},
        )

    ld = LazyDictionary()
    ld["timeframe"] = Timeframe(past_n_days=180)
    ld["pe_trends_df"] = lambda ld: pe_trends_df(ld["timeframe"])
    ld["df"] = lambda ld: data_factory(ld["pe_trends_df"])
    context = {
        "title": "Earnings per sector over time",
        "timeframe": ld["timeframe"],
        "plot_uri": cache_plot(
            f"total-earnings-by-sector:{ld['timeframe'].description}",
            lambda ld: plot(ld["df"]),
            datasets=ld,
        ),
    }
    return render(request, "total_earnings_by_sector.html", context=context)