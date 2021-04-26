"""
Responsible for interaction with worldbank data API and interaction with the user
"""
import re
import io
from collections import defaultdict
from typing import Iterable
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.edit import FormView
from django.http import Http404
from django.urls import reverse
from django.shortcuts import render
import plotnine as p9
from mizani.formatters import date_format
import pandas as pd
from app.data import cache_plot
from app.models import WorldBankIndicators, WorldBankCountry, WorldBankTopic, WorldBankInvertedIndex
from app.forms import WorldBankSCSMForm, WorldBankSCMForm, WorldBankSCMMForm
from app.plots import label_shorten

@login_required
def worldbank_index_view(request):
    return render(request, 'world_bank.html', context={
        "topics":     WorldBankTopic.objects.all().order_by('topic'),
        "indicators": WorldBankIndicators.objects.all(),
        "countries":  WorldBankCountry.objects.all().order_by('name')
    })

def indicator_autocomplete_hits(country_codes: Iterable[str], topic_id: int, as_select_tuples=False) -> tuple:
    assert country_codes is not None
    assert topic_id is not None
    assert len(country_codes) > 0
    assert re.match(r'^\d+$', topic_id)
    for c in country_codes:
        assert re.match(r'^\w{2,4}$', c)
    topic_id = int(topic_id)

    if len(country_codes) == 1: # optimisation to avoid using __in and the performance hit
        records = WorldBankInvertedIndex.objects.filter(topic_id=topic_id, country=next(iter(country_codes))) # cannot index a set, so we use next() instead
    else:
        country_set = set(country_codes) # ensure no dupes
        n_countries = len(country_set)
        #print(f"Topic ID: {topic_id}")
        print(f"Country set: {country_set}")
        records = WorldBankInvertedIndex.objects.filter(topic_id=topic_id, country__in=country_set)
        # show only those indicators FOR WHICH ALL SPECIFIED COUNTRIES are available
        countries_by_indicator = defaultdict(set)
        for r in records:
            countries_by_indicator[r.indicator_id].add(r.country)
        print("Countries for each metric: {}".format(countries_by_indicator))
        acceptable_indicators = set(filter(lambda t: len(countries_by_indicator[t]) == n_countries, countries_by_indicator.keys()))
        #print("Acceptable indicators: {}".format(acceptable_indicators))
        records = records.filter(indicator_id__in=acceptable_indicators)

    hits = []
    seen = set()
    for i in sorted(records, key=lambda i: i.indicator_name):
        key = i.indicator_id
        if key in seen:
            continue
        seen.add(key)
        hits.append(i)
    print("Found {} matching indicators".format(len(hits)))
    current_id = None
    if len(hits) > 0:
        current_id = hits[0].indicator_id
    if as_select_tuples:
        hits = [(i.indicator_id, i.indicator_name) for i in hits]
    return hits, current_id


def fetch_data(indicator: WorldBankIndicators, country_names: Iterable[str], fill_missing=None) -> pd.DataFrame:
    """
    Fetch data from the market_data_cache collection (not to be confused with the market_quote_cache collection)
    and ensure the specified countries are only present in the data (if present). Optionally apply a callable to
    fill in gaps eg. resample
    """
    if indicator is None:
        return None
    with io.BytesIO(indicator.fetch_data()) as fp:
        df = pd.read_parquet(fp)
        if df is not None and len(df) > 0:
            plot_df = df[df["country"].isin(country_names)]
            #print(country_names)
            if len(plot_df) == 0:
                return None
            if fill_missing:
                print(plot_df)
                plot_df.index = pd.to_datetime(plot_df['date'], format='%Y-%m-%d') # avoid callers having to specify 'on' keyword as they may not know which column
                plot_df = fill_missing(plot_df)
                print(plot_df)
            return plot_df
        else:
            return None

def ajax_autocomplete_view(request):
    assert request.method == 'GET'
    topic_id = request.GET.get('topic_id', None)
    country_code = request.GET.get('country', None)
    hits, current_id = indicator_autocomplete_hits([country_code], topic_id)
    return render(request, "autocomplete_indicators.html", context={ 'hits': hits, "current_id": current_id })

def ajax_autocomplete_scm_view(request):
    assert request.method == 'GET'
    topic_id = request.GET.get('topic_id', None)
    country_codes = set(filter(lambda v: len(v) > 0, request.GET.get('country', '').split(",")))
    
    if len(country_codes) > 0:
        hits, current_id = indicator_autocomplete_hits(country_codes, topic_id)
    else:
        hits = []
        current_id = None
    return render(request, "autocomplete_indicators.html", context={ 'hits': hits, "current_id": current_id })

class WorldBankSCSMView(LoginRequiredMixin, FormView): # single-country, single metric (indicator) the easiest to start with
    action_url = '/worldbank/scsm'
    form_class = WorldBankSCSMForm
    template_name = "world_bank_scsm.html"

    def get_form(self, form_class=None):
        if form_class is None:
            form_class = self.get_form_class()
        countries_as_tuples = [(c.country_code, c.name) for c in WorldBankCountry.objects.all().order_by('name')]
        topics_as_tuples = [(t.id, t.topic) for t in WorldBankTopic.objects.all().order_by('topic')]
        kwargs = self.get_form_kwargs()
        data = kwargs.get('data', {})
        selected_country = data.get('country', None)
        selected_topic = data.get('topic', None)
        if selected_topic is not None and selected_country is not None:
            indicators_as_tuples, current_id = indicator_autocomplete_hits([selected_country], selected_topic, as_select_tuples=True)
        else:
            indicators_as_tuples = []
        return form_class(countries_as_tuples, topics_as_tuples, indicators_as_tuples, **kwargs)

    def plot_indicator(self, country: WorldBankCountry, topic: WorldBankTopic, indicator: WorldBankIndicators) -> str:
        def make_plot():
            df = fetch_data(indicator, [country.name], fill_missing=lambda df: df.resample('AS').asfreq())
            if df is None:
                print(f"No usable data/plot for {indicator.name}")
                raise Http404(f"No data for {country.name} found in {indicator.wb_id}")

            plot = ( p9.ggplot(df, p9.aes("date", "metric"))
                    + p9.geom_path(size=1.2) 
                    + p9.theme_classic()
                    + p9.labs(x="", y="Value", title=indicator.name)
                    + p9.theme(figure_size=(12, 6))
                    + p9.scale_y_continuous(labels=label_shorten)
            )
            if "-yearly-" in indicator.tag:
                plot += p9.scale_x_datetime(labels=date_format('%Y'))  # yearly data? if so only print the year on the x-axis
            return plot
        return cache_plot(f"{indicator.wb_id}-{country.name}-scsm-worldbank-plot", make_plot)
            
    def form_valid(self, form):
        #print(form.cleaned_data)
        country = form.cleaned_data.get('country', None)
        topic = form.cleaned_data.get('topic', None)
        indicator = form.cleaned_data.get('indicator', None)
        if country is None or topic is None:
            raise Http404('Both country and topic must be specified for this plot')
        country = WorldBankCountry.objects.filter(country_code=country).first()
        topic = WorldBankTopic.objects.filter(id=topic).first()
        if topic is None:
            raise Http404('No such topic: -{}-'.format(topic))
        
        indicator = WorldBankIndicators.objects.filter(wb_id=indicator).first()
        context = self.get_context_data()
        context.update({
            "title": "World Bank data - single country, single metric",
            "country": country,
            "topic": topic,
            "indicator_autocomplete_uri": reverse('ajax-worldbank-scsm-autocomplete'),
            "indicator_plot_uri": self.plot_indicator(country, topic, indicator),
            "indicator_plot_title": f"Graphic: {indicator.name}",
            "indicator": indicator
        })
        return render(self.request, self.template_name, context=context)

class WorldBankSCMView(LoginRequiredMixin, FormView):
    action_url = "/worldbank/scm"
    form_class = WorldBankSCMForm
    template_name = "world_bank_scm.html"

    def get_form(self, form_class=None):
        if form_class is None:
            form_class = self.get_form_class()
        countries_as_tuples = [(c.country_code, c.name) for c in WorldBankCountry.objects.all().order_by('name')]
        topics_as_tuples = [(t.id, t.topic) for t in WorldBankTopic.objects.all().order_by('topic')]
        kwargs = self.get_form_kwargs()
        data = kwargs.get('data', {})
        selected_countries = data.get('countries', '').split(",")
        selected_topic = data.get('topic', None)
        
        if selected_countries is not None and selected_topic is not None:
            assert isinstance(selected_countries, list)
            indicators_as_tuples, current_id = indicator_autocomplete_hits(selected_countries, selected_topic, as_select_tuples=True)
        else:
            indicators_as_tuples = []
        return form_class(countries_as_tuples, topics_as_tuples, indicators_as_tuples, **kwargs)

    def plot_country_comparison(self, countries: Iterable[str], topic: WorldBankTopic, indicator: WorldBankIndicators) -> p9.ggplot:
        def make_plot():
            df = fetch_data(indicator, countries, fill_missing=lambda df: df.resample('AS').asfreq()) # ensure df has gaps filled in with NAN
            #print(df)
            if df is None:
                print(f"No usable data/plot for {indicator.name}")
                raise Http404(f"No data for {countries} found in {indicator.wb_id}")

            plot = ( p9.ggplot(df, p9.aes("date", "metric", group="country", colour="country"))
                    + p9.geom_path(size=1.2) 
                    + p9.theme_classic()
                    + p9.labs(x="", y="Value", title=indicator.name)
                    + p9.theme(figure_size=(12, 6))
                    + p9.scale_y_continuous(labels=label_shorten)
                    + p9.scale_color_cmap_d()
            )
            if "-yearly-" in indicator.tag:
                plot += p9.scale_x_datetime(labels=date_format('%Y'))  # yearly data? if so only print the year on the x-axis
            return plot
        countries_str = "-".join(countries)
        return cache_plot(f"{indicator.wb_id}-{countries_str}-scm-worldbank-plot", make_plot)

    def form_valid(self, form):
        selected_countries = form.cleaned_data.get('countries', None)
        selected_topic = form.cleaned_data.get('topic', None)
        selected_indicator = form.cleaned_data.get('indicator', None)

        topic = WorldBankTopic.objects.filter(id=selected_topic).first()
        if topic is None:
            raise Http404('No such topic: -{}-'.format(topic))
        countries = WorldBankCountry.objects.filter(country_code__in=selected_countries)
        country_names = [c.name for c in countries]
        indicator = WorldBankIndicators.objects.filter(wb_id=selected_indicator).first()
        context = self.get_context_data()
        context.update({
            "title": "World Bank data - single metric over selected countries",
            "countries": selected_countries,
            "topic": topic,
            "indicator_autocomplete_uri": reverse('ajax-worldbank-scm-autocomplete'),
            "indicator_plot_uri": self.plot_country_comparison(country_names, topic, indicator),
            "indicator_plot_title": f"Graphic: {indicator.name}",
            "indicator": indicator
        })
        return render(self.request, self.template_name, context=context)

class WorldBankSCMMView(LoginRequiredMixin, FormView):
    action_url = "/worldbank/scmm"
    template_name = "world_bank_scmm.html"
    form_class = WorldBankSCMMForm

    def get_form(self, form_class=None):
        if form_class is None:
            form_class = self.get_form_class()
        countries_as_tuples = [(c.country_code, c.name) for c in WorldBankCountry.objects.all().order_by('name')]
        topics_as_tuples = [(t.id, t.topic) for t in WorldBankTopic.objects.all().order_by('topic')]
        kwargs = self.get_form_kwargs()
        data = kwargs.get('data', {})
        selected_country = data.get('country', None)
        selected_topic = data.get('topic', None)
    
        if selected_country is not None and selected_topic is not None:
            indicators_as_tuples, current_id = indicator_autocomplete_hits([selected_country], selected_topic, as_select_tuples=True)
        else:
            indicators_as_tuples = []
        #print(indicators_as_tuples)
        return form_class(countries_as_tuples, topics_as_tuples, indicators_as_tuples, **kwargs)

    def plot_multiple_metrics(self, country: str, topic: WorldBankTopic, indicators: Iterable[WorldBankIndicators]) -> p9.ggplot:
        def make_plot():
            plot_df = None
            has_yearly = False
            n_datasets = 0
            for i in indicators:
                df = fetch_data(i, [country], lambda df: df.resample('AS').asfreq())
                if df is None:
                    continue
                n_datasets += 1
                df['dataset'] = f"{i.name} ({i.wb_id})"
                if "-yearly-" in i.tag:
                    has_yearly = True
                if plot_df is None:
                    plot_df = df
                else:
                    plot_df = plot_df.append(df)

            #print(plot_df)
            plot = ( p9.ggplot(plot_df, p9.aes("date", "metric", group="dataset", colour="dataset"))
                    + p9.geom_path(size=1.2) 
                    + p9.scale_color_cmap_d()
                    + p9.theme_classic()
                    + p9.facet_wrap('~dataset', ncol=1, scales="free_y")
                    + p9.theme(figure_size=(12, n_datasets * 1.5))
                    + p9.scale_y_continuous(labels=label_shorten)
            )
            if has_yearly: # any indicator yearly? if so, then we'll label them all with years only on the plot
                plot += p9.scale_x_datetime(labels=date_format('%Y'))  # yearly data? if so only print the year on the x-axis
            return plot
        indicator_id_str = "-".join([i.wb_id for i in indicators])
        return cache_plot(f"{country}-{indicator_id_str}-scmm-worldbank-plot", make_plot)

    def form_valid(self, form):
        selected_country = form.cleaned_data.get('country', None)
        selected_topic = form.cleaned_data.get('topic', None)
        selected_indicators = form.cleaned_data.get('indicators', [])
        #print(selected_indicators)

        topic = WorldBankTopic.objects.filter(id=selected_topic).first()
        if topic is None:
            raise Http404('No such topic: -{}-'.format(topic))
        country = WorldBankCountry.objects.filter(country_code=selected_country).first()
        indicators = WorldBankIndicators.objects.filter(wb_id__in=set(selected_indicators))
        n_indicators = len(indicators)
        context = self.get_context_data()
        context.update({
            "title": "World Bank data - multiple metrics over single country",
            "country": selected_country,
            "topic": topic,
            "indicator_autocomplete_uri": reverse('ajax-worldbank-scmm-autocomplete'),
            "indicator_plot_uri": self.plot_multiple_metrics(country.name, topic, indicators),
            "indicator_plot_title": f"Graphic: {n_indicators} metrics for {country.name}",
            "indicator_plot_datasets": indicators,
        })
        return render(self.request, self.template_name, context=context)