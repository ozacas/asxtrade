from django.http import HttpResponseRedirect, Http404
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from app.models import (
    user_watchlist,
    validate_user,
    Timeframe,
    validate_stock,
    toggle_watchlist_entry,
    all_etfs,
    Sector,
    all_sector_stocks,
)
from app.data import cache_plot, make_kmeans_cluster_dataframe
from app.plots import user_theme
from app.views.core import show_companies
import pandas as pd
import matplotlib.pyplot as plt
from lazydict import LazyDictionary
import plotnine as p9


@login_required
def show_watched(request):
    validate_user(request.user)
    matching_companies = user_watchlist(request.user)

    timeframe = Timeframe()
    return show_companies(
        matching_companies,
        request,
        timeframe,
        {
            "title": "Stocks you are watching",
            "sentiment_heatmap_title": "Watchlist sentiment heatmap",
        },
    )


@login_required
def cluster_stocks_view(request, stocks: str):
    """
    ref: https://pythonforfinance.net/2018/02/08/stock-clusters-using-k-means-algorithm-in-python/
    """
    validate_user(request.user)
    timeframe = Timeframe(past_n_days=300)
    if stocks == "watchlist":
        asx_codes = user_watchlist(request.user)
    elif stocks == "etfs":
        asx_codes = all_etfs()
    elif stocks.startswith("sector-"):
        sector_id = int(stocks[7:])
        sector = Sector.objects.get(sector_id=sector_id)
        if sector is None or sector.sector_name is None:
            raise Http404("No stocks associated with sector")
        asx_codes = all_sector_stocks(sector.sector_name)
    else:
        raise Http404("Unknown stock list {}".format(stocks))
    chosen_k = 7  # often a reasonable tradeoff

    def elbow_curve_plot(ld: LazyDictionary):
        distortion, _, _, _, _ = make_kmeans_cluster_dataframe(
            timeframe, chosen_k, asx_codes
        )
        fig = plt.figure(figsize=(15, 5))
        plt.plot(range(2, 20), distortion)
        plt.grid(True)
        plt.title("Elbow curve")
        return fig

    def cluster_plot(ld: LazyDictionary):
        _, _, centroids, idx, data_df = make_kmeans_cluster_dataframe(
            timeframe, chosen_k, asx_codes
        )
        centroids_df = pd.DataFrame.from_records(
            centroids, columns=["return", "volatility"]
        )
        plot = (
            p9.ggplot(
                data_df, p9.aes("return", "volatility", colour="factor(cluster_id)")
            )
            + p9.geom_point(size=3)
            + p9.facet_wrap("~cluster_id", ncol=3, scales="free")
        )
        return user_theme(
            plot,
            x_axis_label="Returns (%)",
            y_axis_label="Volatility (%)",
            figure_size=(15, 15),
            subplots_adjust={"hspace": 0.15, "wspace": 0.15},
        )

    stocks_as_str = "-".join(sorted(asx_codes))
    elbow_curve_uri = cache_plot(
        f"{request.user.username}-cluster-{stocks_as_str}-elbow-curve-plot",
        elbow_curve_plot,
    )
    cluster_uri = cache_plot(
        f"{request.user.username}-cluster-{stocks_as_str}-kmeans-cluster-plot",
        cluster_plot,
    )
    context = {
        "elbow_curve_plot_uri": elbow_curve_uri,
        "k": chosen_k,
        "dataset": stocks,
        "n_stocks": len(asx_codes),
        "cluster_plot_uri": cluster_uri,
        "timeframe": timeframe,
    }
    return render(request, "cluster_stocks.html", context=context)


def redirect_to_next(request, fallback_next="/"):
    """
    Call this function in your view once you have deleted some database data: set the 'next' query href
    param to where the redirect should go to. If not specified '/' will be assumed. Not permitted to
    redirect to another site.
    """
    # redirect will trigger a redraw which will show the purchase since next will be the same page
    assert request is not None
    if request.GET is not None:
        next_href = request.GET.get("next", fallback_next)
        assert next_href.startswith("/")  # PARANOIA: must be same origin
        return HttpResponseRedirect(next_href)
    else:
        return HttpResponseRedirect(fallback_next)


@login_required
def toggle_watched(request, stock=None):
    validate_stock(stock)
    validate_user(request.user)
    toggle_watchlist_entry(request.user, stock)
    return redirect_to_next(request)
