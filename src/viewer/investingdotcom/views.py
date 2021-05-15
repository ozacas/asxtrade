from re import S
from django.shortcuts import render
from investingdotcom.forms import CryptoForm, CommodityForm
from investingdotcom.models import get_crypto_prices, get_commodity_prices
from app.models import Timeframe
from app.data import cache_plot
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.edit import FormView
import plotnine as p9
import pandas as pd


class CryptoFormView(LoginRequiredMixin, FormView):
    form_class = CryptoForm
    template_name = "resource_page.html"

    def make_plot(self, df: pd.DataFrame, timeframe: Timeframe) -> p9.ggplot:
        df["Range"] = df["High"] - df["Low"]
        df["Date"] = pd.to_datetime(df.index, format="%Y-%m-%d")
        melted_df = df.melt(
            value_vars=("Close", "Volume", "Range"), id_vars="Date", value_name="value"
        )
        # print(melted_df)
        plot = (
            p9.ggplot(melted_df, p9.aes(x="Date", y="value", group="variable"))
            + p9.geom_line(size=1.3)
            + p9.facet_wrap(
                "~variable",
                ncol=1,
                nrow=melted_df["variable"].nunique(),
                scales="free_y",
            )
            + p9.theme(figure_size=(12, 6))
            + p9.scale_color_cmap_d()
        )
        return plot

    def process_form(self, cleaned_data: dict) -> dict:
        timeframe = Timeframe(past_n_days=cleaned_data["timeframe"])
        crypto_symbol = cleaned_data.get("currency", "BTC")
        crypto_prices = get_crypto_prices(crypto_symbol, timeframe)
        # print(crypto_prices)
        plot_uri = cache_plot(
            f"{crypto_symbol}-{timeframe.description}",
            lambda: self.make_plot(crypto_prices, timeframe),
        )

        return {
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
            lambda: self.make_plot(df, timeframe),
        )
        return {
            "plot_uri": "/png/" + plot_uri,
            "plot_title": f"Commodity prices: {commodity_str} over {timeframe.description}",
        }