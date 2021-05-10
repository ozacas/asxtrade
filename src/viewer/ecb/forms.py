from django import forms
from django.forms.models import model_to_dict
import pandas as pd
from ecb.models import ECBFlow, ECBMetadata, fetch_dataframe


class ECBDynamicDimensionForm(forms.Form):
    """
    Dynamic form which sets up fields to navigate the specified dataflow's dimensions to select the appropriate datapoints for display.
    If there is no current flow then kwargs['dataflow'] is used to initialise things
    """

    df = None

    def __init__(self, current_flow: ECBFlow, **kwargs):
        super().__init__(**kwargs)
        flow_ref = (
            current_flow.name
            if current_flow is not None
            else kwargs.get("dataflow", None)
        )
        dimensions = list(
            [
                model_to_dict(r)
                for r in ECBMetadata.objects.filter(metadata_type="dimension").filter(
                    flow=flow_ref
                )
            ]
        )
        self.df = pd.DataFrame.from_records(dimensions)
        dataset_df = fetch_dataframe(flow_ref)
        assert dataset_df is not None

        for idx, field_name in enumerate(sorted(self.dimensions())):
            values_in_dataset = set(dataset_df[field_name].unique())
            choices = [
                (
                    r.code,
                    r.printable_code[0:80],
                )  # maximum length in chars for the HTML widget is roughly a line length
                for r in sorted(
                    self.df[self.df["column_name"] == field_name].itertuples(),
                    key=lambda r: r.printable_code,
                )
                if r.code in values_in_dataset
            ]
            # print(choices)
            self.fields["dimension_{}".format(field_name)] = forms.ChoiceField(
                choices=choices, required=True, label=field_name
            )
        # print(self.df)

    def dimensions(self):
        return set(self.df["column_name"])
