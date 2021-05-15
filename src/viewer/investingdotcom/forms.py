from django import forms
from worldbank.forms import DynamicChoiceField
from investingdotcom.models import (
    get_cryptocurrencies,
    get_commodities,
    get_bond_countries,
)


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


class BondForm(forms.Form):
    bond_country = forms.ChoiceField(
        choices=(), required=True, label="Country originating bond"
    )
    bond_name = DynamicChoiceField(
        choices=(), required=True, label="Bond"
    )  # choices will be AJAX-initialised and therefore input cant be validated statically
    timeframe = forms.IntegerField(initial=180, required=True, label="Timeframe (days)")

    def __init__(self, countries, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["bond_country"].choices = get_bond_countries(as_choices=True)
