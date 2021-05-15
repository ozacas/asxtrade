from django import forms

from investingdotcom.models import get_cryptocurrencies, get_commodities


class CryptoForm(forms.Form):
    currency = forms.ChoiceField(
        choices=(), required=True, label="Cryptocurrency", initial="BTC"
    )
    timeframe = forms.IntegerField(initial=180, required=True, label="Timeframe (days)")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["currency"].choices = get_cryptocurrencies(as_choices=True)


class CommodityForm(forms.Form):
    commodity = forms.ChoiceField(choices=(), required=True, label="Commodity")
    timeframe = forms.IntegerField(initial=180, required=True, label="Timeframe (days)")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["commodity"].choices = get_commodities(as_choices=True)
