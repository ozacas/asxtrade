from django import forms
from django.core.exceptions import ValidationError
from abs.models import dataflows


class ABSDataflowForm(forms.Form):
    dataflow = forms.ChoiceField(choices=(), required=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["dataflow"].choices = [(i.abs_id, i.name) for i in dataflows()]
