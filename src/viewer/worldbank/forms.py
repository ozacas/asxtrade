from typing import Iterable
from django import forms


class DynamicChoiceField(forms.ChoiceField):
    def valid_value(self, value):  # dont validate here since DOM generated... later
        return True


class DynamicMultiChoiceField(forms.MultipleChoiceField):
    def valid_value(self, value):
        return True


class WorldBankSCSMForm(forms.Form):  # SCSM == single country, single metric
    country = forms.ChoiceField(choices=(), required=True)
    topic = forms.ChoiceField(choices=(), required=True)
    indicator = DynamicChoiceField(
        choices=(), validators=[], required=True
    )  # populated dynamically via AJAX in response to current topic

    def __init__(
        self,
        countries: Iterable[tuple],
        topics: Iterable[tuple],
        indicators: Iterable[tuple],
        **kwargs
    ):
        super(WorldBankSCSMForm, self).__init__(**kwargs)
        self.fields["country"].choices = countries
        # we set initial to ensure AJAX call suceeds
        self.fields["country"].initial = countries[0] if len(countries) > 0 else None
        self.fields["topic"].choices = topics
        self.fields["topic"].initial = topics[0] if len(topics) > 0 else None
        self.fields["indicator"].choices = indicators


class WorldBankSCMForm(forms.Form):  # multi-country, single metric
    countries = forms.MultipleChoiceField(
        choices=(),
        required=True,
        widget=forms.widgets.SelectMultiple(attrs={"size": 12}),
    )
    topic = forms.ChoiceField(choices=(), required=True)
    indicator = DynamicChoiceField(
        choices=(), validators=[], required=True
    )  # only indicators which have ALL selected countries will be shown from the given topic

    def __init__(
        self,
        countries: Iterable[tuple],
        topics: Iterable[tuple],
        indicators: Iterable[tuple],
        **kwargs
    ):
        super(WorldBankSCMForm, self).__init__(**kwargs)
        self.fields["countries"].choices = countries
        self.fields["countries"].initial = countries[0] if len(countries) > 0 else None
        self.fields["topic"].choices = topics
        self.fields["topic"].initial = topics[0] if len(topics) > 0 else None
        self.fields["indicator"].choices = indicators


class WorldBankSCMMForm(
    forms.Form
):  # single country, multiple metrics over the same timeframe
    country = forms.ChoiceField(choices=(), required=True)
    topic = forms.ChoiceField(choices=(), required=True)
    indicators = DynamicMultiChoiceField(
        choices=(),
        validators=[],
        required=True,
        widget=forms.widgets.SelectMultiple(attrs={"size": 12}),
    )  # populated via AJAX

    def __init__(
        self,
        countries: Iterable[tuple],
        topics: Iterable[tuple],
        indicators: Iterable[tuple],
        **kwargs
    ):
        super(WorldBankSCMMForm, self).__init__(**kwargs)
        self.fields["country"].choices = countries
        self.fields["country"].initial = countries[0] if len(countries) > 0 else None
        self.fields["topic"].choices = topics
        self.fields["topic"].initial = topics[0] if len(topics[0]) > 0 else None
        self.fields["indicators"].choices = indicators
