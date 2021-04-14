from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.edit import FormView
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from app.forms import OptimisePortfolioForm, OptimiseSectorForm
from app.messages import info
from app.models import Timeframe, all_sector_stocks, all_etfs, validate_user, selected_cached_stocks_cip, user_watchlist, Sector
from app.views.core import show_companies
from app.analysis import detect_outliers, optimise_portfolio

@login_required
def show_watchlist_outliers(request, n_days=30):
    validate_user(request.user)
    stocks = user_watchlist(request.user)
    return show_outliers(request, stocks, n_days=n_days)


def show_outliers(request, stocks, n_days=30, extra_context=None):
    assert stocks is not None
    assert n_days is not None  # typically integer, but desired_dates() is polymorphic
    timeframe = Timeframe(past_n_days=n_days)
    cip = selected_cached_stocks_cip(stocks, timeframe)
    outliers = detect_outliers(stocks, cip)
    extra_context = {
        "title": "Unusual stock behaviours: {}".format(timeframe.description),
        "sentiment_heatmap_title": "Outlier stocks: sentiment",
    }
    return show_companies(
        outliers,
        request,
        timeframe,
        extra_context,
    )


class OptimisedWatchlistView(
        LoginRequiredMixin,
        FormView
):
    action_url = '/show/optimized/watchlist/'
    template_name = 'optimised_view.html'
    form_class = OptimisePortfolioForm
    results = None # specified when valid form submitted
    stock_title = "Watchlist"

    def get_context_data(self, **kwargs):
        ret = super().get_context_data(**kwargs)
        if self.results is not None:
            (
                cleaned_weights,
                performance,
                efficient_frontier_plot,
                correlation_plot,
                messages,
                title,
                portfolio_cost,
                leftover_funds,
                n_stocks,
            ) = self.results
            for msg in messages:
                info(self.request, msg)
            total_pct_cw = sum(map(lambda t: t[1], cleaned_weights.values())) * 100.0
            #print(cleaned_weights)
            total_profit = sum(map(lambda t: t[5], cleaned_weights.values()))
            ret.update({
                "timeframe": self.timeframe,
                "cleaned_weights": cleaned_weights,
                "algo": title,
                "portfolio_performance_dict": performance,
                "efficient_frontier_plot_uri": efficient_frontier_plot,
                "correlation_plot_uri": correlation_plot,
                "portfolio_cost": portfolio_cost,
                "total_cleaned_weight_pct": total_pct_cw,
                "total_profit_aud": total_profit,
                "leftover_funds": leftover_funds,
                "stock_selector": self.stock_title,
                "n_stocks_considered": n_stocks,
                "n_stocks_in_portfolio": len(cleaned_weights.keys())
            })
        return ret

    def stocks(self):
        return list(user_watchlist(self.request.user))

    def get_form(self, form_class=None):
        if form_class is None:
            form_class = self.get_form_class()
        return form_class(sorted(self.stocks()), **self.get_form_kwargs())

    def form_valid(self, form):
        exclude = form.cleaned_data['excluded_stocks']
        n_days = form.cleaned_data['n_days']
        algo = form.cleaned_data['method']
        portfolio_cost = form.cleaned_data['portfolio_cost']
        exclude_price = form.cleaned_data.get('exclude_price', None)
        stocks = self.stocks()

        if exclude is not None:
            if isinstance(exclude, str):
                exclude = exclude.split(",")
            stocks = set(stocks).difference(exclude)
        
        self.timeframe = Timeframe(past_n_days=n_days)
        self.results = optimise_portfolio(stocks,
                                  self.timeframe,
                                  algo=algo,
                                  total_portfolio_value=portfolio_cost, 
                                  exclude_price=exclude_price)
        return render(self.request, self.template_name, self.get_context_data())

optimised_watchlist_view = OptimisedWatchlistView.as_view()

class OptimisedSectorView(OptimisedWatchlistView):
    sector = None # initialised by get_queryset()
    action_url = '/show/optimized/sector/'
    stock_title = "Sector"
    form_class = OptimiseSectorForm

    def stocks(self):
        if self.sector is None:
            self.sector = 'Information Technology'
        return sorted(all_sector_stocks(self.sector))

    def get_form_kwargs(self):
        """Permit the user to provide initial form value for sector as a HTTP GET query parameter"""
        ret = super().get_form_kwargs()
        sector = self.request.GET.get('sector', None)
        if sector:
            ret.update({'sector': sector})
        return ret

    def form_valid(self, form):
        self.sector = form.cleaned_data['sector']
        self.stock_title = "{} sector".format(self.sector)
        return super().form_valid(form)

optimised_sector_view = OptimisedSectorView.as_view()

class OptimisedETFView(OptimisedWatchlistView):
    action_url = '/show/optimized/etfs/'
    stock_title = "ETFs"

    def stocks(self):
        return sorted(all_etfs())

optimised_etf_view = OptimisedETFView.as_view()

@login_required
def show_sector_outliers(request, sector_id=None, n_days=30):
    validate_user(request.user)
    assert isinstance(sector_id, int) and sector_id > 0

    stocks = all_sector_stocks(Sector.objects.get(sector_id=sector_id).sector_name)
    return show_outliers(request, stocks, n_days=n_days)
