"""
Responsible for implementing user search for finding companies of interest
"""
from collections import defaultdict
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.list import MultipleObjectTemplateResponseMixin
from django.views.generic.edit import FormView
from django.shortcuts import render
import pandas as pd
from pandas.core.base import NoNewAttributesMixin
from app.mixins import SearchMixin
from app.data import cache_plot, calc_ma_crossover_points, timeframe_end_performance
from app.messages import info
from app.forms import (
    DividendSearchForm,
    MomentumSearchForm,
    MoverSearchForm,
    CompanySearchForm,
    MarketCapSearchForm,
    SectorSentimentSearchForm,
    FinancialMetricSearchForm,
)
from app.models import (
    Timeframe,
    Quotation,
    all_stocks,
    latest_quotation_date,
    stocks_by_sector,
    user_watchlist,
    valid_quotes_only,
    latest_quote,
    Sector,
    all_sector_stocks,
    company_prices,
    find_movers,
    find_named_companies,
    selected_cached_stocks_cip,
    CompanyFinancialMetric,
)
from app.data import make_sector_performance_dataframe
from app.views.core import show_companies
from app.plots import plot_sector_performance, plot_boxplot_series
from app.messages import warning
from lazydict import LazyDictionary
from plotnine.layer import Layers
from cachetools import LRUCache


class DividendYieldSearch(
    SearchMixin,
    LoginRequiredMixin,
    MultipleObjectTemplateResponseMixin,
    FormView,
):
    form_class = DividendSearchForm
    template_name = "search_form.html"  # generic template, not specific to this view
    action_url = "/search/by-yield"
    ordering = (
        "asx_code"  # default order must correspond to default JS code in template
    )
    timeframe = Timeframe(past_n_days=30)
    qs = None
    query_cache = LRUCache(maxsize=2)

    def additional_context(self, context):
        """
        Return the additional fields to be added to the context by render_to_response(). Subclasses
        should override this rather than the template design pattern implementation of render_to_response()
        """
        assert context is not None
        if self.timeframe is not None:
            info(
                self.request,
                f"Unless stated otherwise, timeframe is {self.timeframe.description}",
            )
        return {
            "title": "Find by dividend yield or P/E",
            "timeframe": self.timeframe,
            "sentiment_heatmap_title": "Matching stock heatmap",
            "n_top_bottom": 20,
        }

    def render_to_response(self, context, **kwargs):
        """
        Invoke show_companies()
        """
        assert kwargs is not None
        context.update(self.additional_context(context))

        return show_companies(  # will typically invoke show_companies() to share code across all views
            self.qs,
            self.request,
            self.timeframe,
            context,
            template_name=self.template_name,
        )

    def recalc_queryset(self, **kwargs):
        if kwargs == {}:
            return Quotation.objects.none()

        as_at = latest_quotation_date("ANZ")
        min_yield = kwargs.get("min_yield") if "min_yield" in kwargs else 0.0
        max_yield = kwargs.get("max_yield") if "max_yield" in kwargs else 10000.0
        results = (
            Quotation.objects.filter(fetch_date=as_at)
            .filter(annual_dividend_yield__gte=min_yield)
            .filter(annual_dividend_yield__lte=max_yield)
        )

        if "min_pe" in kwargs:
            results = results.filter(pe__gte=kwargs.get("min_pe"))
        if "max_pe" in kwargs:
            results = results.filter(pe__lt=kwargs.get("max_pe"))
        if "min_eps_aud" in kwargs:
            results = results.filter(eps__gte=kwargs.get("min_eps_aud"))
        return results

    def get_queryset(self, **kwargs):
        """
        DO NOT override this method, use recalc_queryset() instead so that you get caching behaviour to speed page navigation
        """
        cache_key = (
            self.action_url + "-" + "-".join([f"{k}={v}" for k, v in kwargs.items()])
        )
        if cache_key in self.query_cache:
            print(f"Using cached queryset: {cache_key}")
            self.qs = self.query_cache[cache_key]
            return self.qs
        else:
            self.qs = self.recalc_queryset(**kwargs)
            self.query_cache[cache_key] = self.qs
            return self.qs


dividend_search = DividendYieldSearch.as_view()


class MoverSearch(DividendYieldSearch):
    form_class = MoverSearchForm
    action_url = "/search/movers"

    def additional_context(self, context):
        d = super().additional_context(context)
        d.update(
            {
                "title": "Find companies exceeding threshold over timeframe",
                "sentiment_heatmap_title": "Heatmap for moving stocks",
            }
        )
        return d

    def recalc_queryset(self, **kwargs):
        if any(
            [kwargs == {}, "threshold" not in kwargs, "timeframe_in_days" not in kwargs]
        ):
            return Quotation.objects.none()
        threshold_percentage = kwargs.get("threshold")
        self.timeframe = Timeframe(past_n_days=kwargs.get("timeframe_in_days", 30))

        df = find_movers(
            threshold_percentage,
            self.timeframe,
            kwargs.get("show_increasing", False),
            kwargs.get("show_decreasing", False),
            kwargs.get("max_price", None),
            field=kwargs.get("metric", "change_in_percent"),
        )
        # print(df)

        ret, _ = latest_quote(tuple(df.index))
        return ret


mover_search = MoverSearch.as_view()


class CompanySearch(DividendYieldSearch):
    DEFAULT_SECTOR = "Communication Services"
    ld = LazyDictionary()
    form_class = CompanySearchForm
    action_url = "/search/by-company"

    def additional_context(self, context):
        ld = self.ld
        sector = ld.get(
            "sector", self.DEFAULT_SECTOR
        )  # default to Comms. Services if not specified'
        sector_id = ld.get("sector_id", None)
        return {
            "title": "Find by company details",
            "sentiment_heatmap_title": "Heatmap for matching stocks",
            # to highlight top10/bottom10 bookmarks correctly
            "sector_name": sector,
            "sector_id": sector_id,
            "sentiment_heatmap_title": "{} sector sentiment".format(sector),
            "sector_performance_plot_uri": ld.get("sector_performance_plot", None),
            "timeframe_end_performance": timeframe_end_performance(ld),
        }

    def sector_performance(self, ld: LazyDictionary) -> str:
        sector = ld.get("sector")
        if len(ld['sector_performance_df']) < 1:
            return None
        return cache_plot(
            f"{sector}-sector-performance",
            lambda ld: plot_sector_performance(ld["sector_performance_df"], sector),
            datasets=ld,
            dont_cache=True,
        )

    def filter_top_bottom(
        self, ld: LazyDictionary, wanted_stocks, report_top_n, report_bottom_n
    ) -> set:
        if report_top_n is not None or report_bottom_n is not None:
            cip_sum = self.ld["cip_df"].transpose().sum().to_frame(name="percent_cip")

            # print(cip_sum)
            top_N = (
                set(cip_sum.nlargest(report_top_n, "percent_cip").index)
                if report_top_n is not None
                else set()
            )
            bottom_N = (
                set(cip_sum.nsmallest(report_bottom_n, "percent_cip").index)
                if report_bottom_n is not None
                else set()
            )
            wanted_stocks = top_N.union(bottom_N)
        else:
            return wanted_stocks

    def recalc_queryset(self, **kwargs):
        if kwargs == {} or not any(
            ["name" in kwargs, "activity" in kwargs, "sector" in kwargs]
        ):
            return Quotation.objects.none()

        wanted_name = kwargs.get("name", "")
        wanted_activity = kwargs.get("activity", "")
        if len(wanted_name) > 0 or len(wanted_activity) > 0:
           matching_companies = find_named_companies(wanted_name, wanted_activity)
        else:
           matching_companies = all_stocks()
        sector = kwargs.get("sector", self.DEFAULT_SECTOR)
        sector_id = int(Sector.objects.get(sector_name=sector).sector_id)
        sector_stocks = all_sector_stocks(sector)
        if kwargs.get("sector_enabled", False):
           matching_companies = matching_companies.intersection(sector_stocks)
        print("Found {} companies matching: name={} or activity={}".format(len(matching_companies), wanted_name, wanted_activity))
 
        report_top_n = kwargs.get("report_top_n", None)
        report_bottom_n = kwargs.get("report_bottom_n", None)
        self.timeframe = Timeframe(past_n_days=90)
        ld = LazyDictionary()
        ld["sector"] = sector
        ld["sector_id"] = sector_id
        ld["sector_companies"] = sector_stocks
        if len(matching_companies) > 0:
            ld["cip_df"] = selected_cached_stocks_cip(matching_companies, self.timeframe)
        else:
            ld["cip_df"] = pd.DataFrame()
        ld["sector_performance_df"] = lambda ld: make_sector_performance_dataframe(
            ld["cip_df"], ld["sector_companies"]
        )
        ld["sector_performance_plot"] = lambda ld: self.sector_performance(ld)
        self.ld = ld
        wanted_stocks = self.filter_top_bottom(
            ld, matching_companies, report_top_n, report_bottom_n
        )

        print("Requesting valid quotes for {} stocks".format(len(wanted_stocks)))
        quotations_as_at, actual_mrd = valid_quotes_only(
            "latest", ensure_date_has_data=True
        )
        ret = quotations_as_at.filter(asx_code__in=wanted_stocks)
        if len(ret) < len(wanted_stocks):
            got = set([q.asx_code for q in self.qs.all()]) if self.qs else set()
            missing_stocks = wanted_stocks.difference(got)
            warning(
                self.request,
                f"could not obtain quotes for all stocks as at {actual_mrd}: {missing_stocks}",
            )

        print("Showing results for {} companies".format(len(matching_companies)))
        ret, _ = latest_quote(tuple(matching_companies))
        return ret


company_search = CompanySearch.as_view()


class FinancialMetricSearchView(SearchMixin, LoginRequiredMixin, FormView):
    form_class = FinancialMetricSearchForm
    template_name = "search_form.html"
    action_url = "/search/by-metric"

    def get_queryset(self, **kwargs):
        """
        Invoke show_companies()
        """
        wanted_metric = kwargs.get("metric", None)
        wanted_value = kwargs.get("amount", None)
        wanted_unit = kwargs.get("unit", "")
        wanted_relation = kwargs.get("relation", ">=")

        mult = 1 if not wanted_unit.endswith("(M)") else 1000 * 1000
        print(f"{wanted_metric} {wanted_value} {mult} {wanted_relation}")

        if all([wanted_metric, wanted_value, wanted_relation]):
            matching_records = CompanyFinancialMetric.objects.filter(name=wanted_metric)
            if wanted_relation == "<=":
                matching_records = matching_records.filter(
                    value__lte=wanted_value * mult
                )
            else:
                matching_records = matching_records.filter(
                    value__gte=wanted_value * mult
                )
            matching_stocks = set([m.asx_code for m in matching_records])
            # its possible that some stocks have been delisted at the latest date, despite the financials being a match.
            # They will be shown if the ASX data endpoint still returns data as at the latest quote date
        else:
            matching_stocks = Quotation.objects.none()
        return matching_stocks  # will be assigned to self.object_list by superclass

    def render_to_response(self, context):
        context.update(
            {
                "title": "Find companies by financial metric",
                "sentiment_heatmap_title": "Matching stock sentiment",
            }
        )
        warning(
            self.request,
            "Due to experimental data ingest, results may be wrong/inaccurate/misleading. Use at own risk",
        )
        return show_companies(
            self.object_list,  # ie. return result from self.get_queryset()
            self.request,
            Timeframe(past_n_days=30),
            context,
            self.template_name,
        )


financial_metric_search = FinancialMetricSearchView.as_view()


class MarketCapSearch(MoverSearch):
    action_url = "/search/market-cap"
    form_class = MarketCapSearchForm

    def additional_context(self, context):
        return {
            "title": "Find companies by market capitalisation",
            "sentiment_heatmap_title": "Heatmap for matching market cap stocks",
        }

    def recalc_queryset(self, **kwargs):
        # identify all stocks which have a market cap which satisfies the required constraints
        quotes_qs, most_recent_date = latest_quote(None)
        min_cap = kwargs.get("min_cap", 1)
        max_cap = kwargs.get("max_cap", 1000)
        quotes_qs = (
            quotes_qs.exclude(market_cap__lt=min_cap * 1000 * 1000)
            .exclude(market_cap__isnull=True)
            .exclude(market_cap__gt=max_cap * 1000 * 1000)
            .exclude(suspended__isnull=True)
        )
        info(
            self.request,
            "Found {} quotes, as at {}, satisfying market cap criteria".format(
                quotes_qs.count(), most_recent_date
            ),
        )
        return quotes_qs


market_cap_search = MarketCapSearch.as_view()


class ShowRecentSectorView(LoginRequiredMixin, FormView):
    template_name = "recent_sector_performance.html"
    form_class = SectorSentimentSearchForm
    action_url = "/show/recent_sector_performance"

    def form_valid(self, form):
        sector = form.cleaned_data.get("sector", "Communication Services")
        norm_method = form.cleaned_data.get("normalisation_method", None)
        n_days = form.cleaned_data.get("n_days", 30)
        ld = LazyDictionary()
        ld["stocks"] = lambda ld: all_sector_stocks(sector)
        ld["timeframe"] = Timeframe(past_n_days=n_days)
        ld["cip_df"] = lambda ld: selected_cached_stocks_cip(
            ld["stocks"], ld["timeframe"]
        )

        context = self.get_context_data()

        def winner_results(df: pd.DataFrame) -> list:
            # compute star performers: those who are above the mean on a given day counted over all days
            count = defaultdict(int)
            avg = df.mean(axis=0)
            for col in df.columns:
                winners = df[df[col] > avg[col]][col]
                for winner in winners.index:
                    count[winner] += 1
            results = []
            for asx_code, n_wins in count.items():
                x = df.loc[asx_code].sum()
                # avoid "dead cat bounce" stocks which fall spectacularly and then post major increases in percentage terms
                if x > 0.0:
                    results.append((asx_code, n_wins, x))
            return list(reversed(sorted(results, key=lambda t: t[2])))

        context.update(
            {
                "title": "Past {} day sector performance: box plot trends".format(
                    n_days
                ),
                "n_days": n_days,
                "sector": sector,
                "plot_uri": cache_plot(
                    f"{sector}-recent-sector-view-{ld['timeframe'].description}-{norm_method}",
                    lambda ld: plot_boxplot_series(
                        ld["cip_df"], normalisation_method=norm_method
                    ),
                    datasets=ld,
                ),
                "winning_stocks": winner_results(ld["cip_df"]),
            }
        )
        return render(self.request, self.template_name, context)


show_recent_sector = ShowRecentSectorView.as_view()


class MomentumSearch(DividendYieldSearch):
    """
    Search for momentum related signals by finding cross over points between 20-day moving average and 200 day moving average.
    We try to provide a warm-up period of data (depending on what it is in the database) so that the user-requested period has good data.
    """

    form_class = MomentumSearchForm
    action_url = "/search/momentum-change"
    template_name = "search_form.html"

    def additional_context(self, context):
        ret = super().additional_context(context)
        ret.update(
            {
                "title": "Momentum Search",
                "sentiment_heatmap_title": "Momentum stock sentiment",
            }
        )
        return ret

    def recalc_queryset(self, **kwargs):
        n_days = kwargs.get("n_days", 30)
        what_to_search = kwargs.get("what_to_search")
        period1 = kwargs.get("period1", 20)
        period2 = kwargs.get("period2", 200)
        if what_to_search == "all_stocks":
            stocks_to_consider = all_stocks()
        elif what_to_search == "watchlist":
            stocks_to_consider = user_watchlist(self.request.user)
        else:
            #print(what_to_search)
            stocks_to_consider = all_sector_stocks(what_to_search)
       
        matching_stocks = set()
        self.timeframe = Timeframe(past_n_days=n_days)

        assert period2 > period1
        df = company_prices(
            stocks_to_consider, Timeframe(past_n_days=n_days + period2), transpose=False
        )
        # print(df)
        wanted_dates = set(self.timeframe.all_dates())
        for s in filter(lambda asx_code: asx_code in df.columns, stocks_to_consider):
            last_price = df[s]
            # we filter now because it is after the warm-up period for MA200....
            ma20 = last_price.rolling(period1).mean().filter(items=wanted_dates, axis=0)
            ma200 = (
                last_price.rolling(period2, min_periods=min([50, 3 * period1]))
                .mean()
                .filter(items=wanted_dates, axis=0)
            )

            matching_dates = set(
                [xo[1] for xo in calc_ma_crossover_points(ma20, ma200)]
            )
            if len(matching_dates.intersection(wanted_dates)) > 0:
                matching_stocks.add(s)
        return matching_stocks


momentum_change_search = MomentumSearch.as_view()
