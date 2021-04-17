from django.http import HttpResponseRedirect, Http404
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from app.models import (user_watchlist, validate_user, Timeframe, validate_stock, toggle_watchlist_entry, company_prices)
from app.data import cache_plot
from app.views.core import show_companies
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,vq
from sklearn.cluster import KMeans
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
            "sentiment_heatmap_title": "Watchlist stocks sentiment: {}".format(timeframe.description),
        }
    )

@login_required
def cluster_stocks_view(request, stocks: str):
    """
    ref: https://pythonforfinance.net/2018/02/08/stock-clusters-using-k-means-algorithm-in-python/
    """
    validate_user(request.user)
    timeframe = Timeframe(past_n_days=300)
    if stocks == "watchlist":
        stocks = user_watchlist(request.user)
    else:
        raise Http404('Unknown stock list {}'.format(stocks))
    chosen_k = 7 # often a reasonable tradeoff

    def data_factory():
        prices_df = company_prices(stocks, timeframe, fields="last_price")
        #print(prices_df)
        s1 = prices_df.pct_change().mean() * 252
        s2 = prices_df.pct_change().std() * math.sqrt(252.0)
        #print(s1)
        data_df = pd.DataFrame.from_dict({'return': s1, 'volatility': s2 })
        #print(data_df)
        data = np.asarray([np.asarray(data_df['return']),np.asarray(data_df['volatility'])]).T
        distortion = []
        for k in range(2, 20):
            k_means = KMeans(n_clusters=k)
            k_means.fit(data)
            distortion.append(k_means.inertia_)
        # computing K-Means with K = 5 (5 clusters)
        centroids,_ = kmeans(data, chosen_k)
        # assign each sample to a cluster
        idx,_ = vq(data, centroids)
        return distortion, chosen_k, centroids, idx, data_df

    def elbow_curve_plot():
        distortion, _, _, _, _ = data_factory()
        fig = plt.figure(figsize=(15, 5))
        plt.plot(range(2, 20), distortion)
        plt.grid(True)
        plt.title('Elbow curve')
        return fig
    
    def cluster_plot():
        _, _, centroids, idx, data_df = data_factory()
        data_df["cluster_id"] = idx
        #data_df = data_df[data_df["return"] < 50.0]
        return (p9.ggplot(data_df, p9.aes("return", "volatility", colour="factor(cluster_id)"))
                + p9.geom_point(show_legend=False)
                + p9.facet_wrap('~cluster_id', ncol=3, scales="free")
                + p9.theme(subplots_adjust={'hspace': 0.25, 'wspace': 0.25}, 
                           axis_text_x=p9.element_text(size=6), 
                           axis_text_y=p9.element_text(size=6))
        )

    elbow_curve_uri = cache_plot(f"{request.user.username}-watchlist-cluster-stocks-elbow-curve-plot", elbow_curve_plot)
    cluster_uri = cache_plot(f"{request.user.username}-watchlist-cluster-stocks-kmeans-cluster-plot", cluster_plot)
    context = {
        "elbow_curve_plot_uri": elbow_curve_uri,
        "k": chosen_k,
        "n_stocks": len(stocks),
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
