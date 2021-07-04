from re import S
from django.shortcuts import render
from investingdotcom.forms import CryptoForm, CommodityForm, BondForm
from investingdotcom.models import (
    get_bond_countries,
    get_bonds_for_country,
    get_crypto_prices,
    get_commodity_prices,
    get_bond_prices,
)
from app.models import Timeframe, validate_user
from app.data import cache_plot
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.views.generic.edit import FormView
import plotnine as p9
import pandas as pd
from app.plots import user_theme


class CryptoFormView(LoginRequiredMixin, FormView):
    form_class = CryptoForm
    template_name = "resource_page.html"
    value_vars = ("Close", "Volume", "Range")

    def make_plot(self, df: pd.DataFrame, timeframe: Timeframe) -> p9.ggplot:
        # print(df)
        col_names = set(df.columns)
        if "High" in col_names and "Low" in col_names:
            df["Range"] = df["High"] - df["Low"]
        df["Date"] = pd.to_datetime(df.index, format="%Y-%m-%d")
        melted_df = df.melt(
            value_vars=self.value_vars, id_vars="Date", value_name="value"
        )
        # print(melted_df)
        plot = (
            p9.ggplot(
                melted_df,
                p9.aes(x="Date", y="value", group="variable", color="variable"),
            )
            + p9.geom_line(size=1.3)
            + p9.facet_wrap(
                "~variable",
                ncol=1,
                nrow=melted_df["variable"].nunique(),
                scales="free_y",
            )
        )
        return user_theme(plot)

    def process_form(self, cleaned_data: dict) -> dict:
        timeframe = Timeframe(past_n_days=cleaned_data["timeframe"])
        crypto_symbol = cleaned_data.get("currency", "BTC")
        crypto_prices = get_crypto_prices(crypto_symbol, timeframe)
        # print(crypto_prices)
        plot_uri = cache_plot(
            f"{crypto_symbol}-{timeframe.description}",
            lambda ld: self.make_plot(crypto_prices, timeframe),
        )

        return {
            "title": "Visualize cryptocurrency prices over time",
            "plot_uri": "/png/" + plot_uri,
            "plot_title": f"{crypto_symbol} over {timeframe.description}",
        }

    def form_valid(self, form):
        context = self.get_context_data()
        context.update(self.process_form(form.cleaned_data))
        return render(self.request, self.template_name, context=context)


class CommodityFormView(CryptoFormView):
    form_class = CommodityForm

    def process_form(self, cleaned_data):
        timeframe = Timeframe(past_n_days=cleaned_data.get("timeframe", 180))
        commodity_str = cleaned_data.get("commodity", "Gold")
        df = get_commodity_prices(commodity_str, timeframe)
        plot_uri = cache_plot(
            f"{commodity_str}-{timeframe.description}",
            lambda ld: self.make_plot(df, timeframe),
        )
        return {
            "title": "Visualize commodity prices over time",
            "plot_uri": "/png/" + plot_uri,
            "plot_title": f"Commodity prices: {commodity_str} over {timeframe.description}",
        }


class BondFormView(CryptoFormView):
    form_class = BondForm
    template_name = "bond_page.html"
    value_vars = ("Close", "Range")

    def get_form(self, form_class=None):
        if form_class is None:
            form_class = self.get_form_class()
        countries = get_bond_countries()
        kwargs = self.get_form_kwargs()
        return form_class(countries, **kwargs)

    def process_form(self, cleaned_data):
        timeframe = Timeframe(past_n_days=cleaned_data.get("timeframe", 180))
        bond = cleaned_data.get("bond_name", None)
        df = get_bond_prices(bond, timeframe)
        plot_uri = cache_plot(
            f"{bond}-{timeframe.description}",
            lambda ld: self.make_plot(df, timeframe),
        )
        return {
            "title": "Visualise bond yields by country",
            "plot_uri": "/png/" + plot_uri,
            "plot_title": f"Bond prices: {bond} over {timeframe.description}",
        }


@login_required
def ajax_country_bond_autocomplete(request):
    validate_user(request.user)
    assert request.method == "GET"
    country = request.GET.get("country", None)

    bonds = get_bonds_for_country(country)
    assert isinstance(bonds, pd.DataFrame)
    hits = []
    # print(bonds)
    for i, s in bonds.iterrows():
        assert s["country"] == country
        hits.append({"id": s["name"], "name": s["full_name"]})

    return render(
        request,
        "country_bond_autocomplete_hits.html",
        context={"hits": hits, "current_id": country},
    )