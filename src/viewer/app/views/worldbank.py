"""
Responsible for interaction with worldbank data API and interaction with the user
"""
import re
import io
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
from app.forms import WorldBankSCSMForm

@login_required
def worldbank_index_view(request):
    return render(request, 'world_bank.html', context={
        "topics":     WorldBankTopic.objects.all().order_by('topic'),
        "indicators": WorldBankIndicators.objects.all(),
        "countries":  WorldBankCountry.objects.all().order_by('name')
    })

def indicator_autocomplete_hits(country_code: str, topic_id: int):
    assert country_code is not None
    assert topic_id is not None
    assert re.match(r'^\d+$', topic_id)
    assert re.match(r'^\w{1,4}$', country_code)
    topic_id = int(topic_id)

    hits = [i for i in sorted(WorldBankInvertedIndex.objects.filter(topic_id=topic_id, country=country_code), 
                              key=lambda i: i.indicator_name)]
    print("Found {} matching indicators".format(len(hits)))
    current_id = None
    if len(hits) > 0:
        current_id = hits[0].indicator_id
    return hits, current_id

def ajax_autocomplete_indicator_view(request):
    assert request.method == 'GET'
    topic_id = request.GET.get('topic_id', None)
    country_code = request.GET.get('country', None)
    hits, current_id = indicator_autocomplete_hits(country_code, topic_id)
    return render(request, "autocomplete_indicators.html", context={ 'hits': hits, "current_id": current_id })

class WorldBankSCSMView(LoginRequiredMixin, FormView):
    action_url = '/worldbank/indicators'
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
        hits, current_id = indicator_autocomplete_hits(selected_country, selected_topic)
        indicators_as_tuples = [(i.indicator_id, i.indicator_name) for i in hits]
        return form_class(countries_as_tuples, topics_as_tuples, indicators_as_tuples, **self.get_form_kwargs())

    def fetch_data(self, indicator: WorldBankIndicators, country_name: str) -> pd.DataFrame:
        if indicator is None:
            return None
        with io.BytesIO(indicator.fetch_data()) as fp:
            df = pd.read_parquet(fp)
            if df is not None and len(df) > 0:
                plot_df = df[df["country"] == country_name]
                if len(plot_df) == 0:
                    return None
                return plot_df
            else:
                return None

    def plot_indicator(self, country: WorldBankCountry, topic: WorldBankTopic, indicator: WorldBankIndicators) -> str:
        df = self.fetch_data(indicator, country.name)
        if df is None:
            print(f"No usable data/plot for {indicator.name}")
            raise Http404(f"No data for {country.name} found in {indicator.wb_id}")

        plot = ( p9.ggplot(df, p9.aes("date", "metric"))
                + p9.geom_line(size=1.2) 
                + p9.theme_classic()
                + p9.labs(x="", y="Value", title=indicator.name)
                + p9.theme(figure_size=(12, 6))
        )
        if "-yearly-" in indicator.tag:
            plot += p9.scale_x_datetime(labels=date_format('%Y'))  # yearly data? if so only print the year on the x-axis
        return cache_plot(f"{indicator.wb_id}-worldbank-plot", lambda: plot)
            
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
            "indicator_autocomplete_uri": reverse('ajax-worldbank-indicator-autocomplete-view'),
            "indicator_plot_uri": self.plot_indicator(country, topic, indicator),
            "indicator_plot_title": f"Graphic: {indicator.name}",
            "indicator": indicator
        })
        return render(self.request, "world_bank_scsm.html", context=context)

