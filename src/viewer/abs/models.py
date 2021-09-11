import django.db.models as model
from djongo.models import ObjectIdField, DjongoManager
from django.conf import settings
import json
import requests
from typing import Iterable
import pandas as pd
import pandasdmx as sdmx


class ABSDataCache(model.Model):
    """
    Similar to, but separate from, app.MarketQuoteCache, this keeps track of pandas dataframes (parquet format) which have
    been downloaded, cleaned and ingested into MongoDB
    """

    size_in_bytes = model.IntegerField()
    status = model.TextField()
    tag = model.TextField()
    dataframe_format = model.TextField()
    field = model.TextField()
    last_updated = model.DateTimeField()
    market = model.TextField()
    n_days = model.IntegerField()
    n_stocks = model.IntegerField()
    sha256 = model.TextField()
    _id = ObjectIdField()
    scope = model.TextField()
    dataframe = model.BinaryField()

    objects = DjongoManager()

    class Meta:
        db_table = "abs_data_cache"


class ABSInvertedIndex(model.Model):
    id = ObjectIdField(primary_key=True, db_column="_id")
    min_date = model.DateTimeField()
    max_date = model.DateTimeField()
    n_attributes = model.IntegerField()
    last_updated = model.DateTimeField(auto_now_add=True)
    tag = model.TextField()
    dataset = model.TextField()
    name = model.TextField()
    scope = model.TextField()  # always 'abs' for now

    objects = DjongoManager()

    class Meta:
        db_table = "abs_inverted_index"


class ABSDataflow(model.Model):
    abs_id = model.TextField()
    name = model.TextField()
    objects = DjongoManager()


class ABSHeadlineDataRecord(model.Model):
    dataflow = model.TextField()
    idx = model.IntegerField(blank=False, null=False)
    time_period = model.TextField(null=False, blank=False)
    variable = model.TextField(null=False, blank=False)
    val = model.TextField()

    objects = DjongoManager()

    class Meta:
        db_table = "abs_headline_data"


def update_datapoints(df: pd.DataFrame, dataflow: str) -> None:
    assert dataflow is not None and len(dataflow) > 0
    df = df.rename(
        columns={"TIME_PERIOD": "time_period"}
    )  # for django model compatibility
    cols = set(df.columns)
    df["dataflow"] = dataflow
    df["idx"] = df.index
    id_cols = set(["time_period", "dataflow", "idx"])
    value_cols = cols.difference(id_cols)
    ABSHeadlineDataRecord.objects.filter(
        time_period__in=df["time_period"].unique(), dataflow=dataflow
    ).delete()
    df = df.melt(id_vars=id_cols, value_vars=value_cols, value_name="val")
    # print(df)
    n = 0
    for idx, d in enumerate(df.to_dict("records")):
        n += 1
        ABSHeadlineDataRecord.objects.create(**d)

    print(f"Updated {n} ABS data records.")


def data(dataflow_id: str, abs_api_key: str = None) -> pd.DataFrame:
    assert len(dataflow_id) > 0
    if abs_api_key is None:
        abs_api_key = settings.ABS_API_KEY
    headers = {
        # "Accept": "application/json", # sdmx api doesnt support application/json, but luckily both ABS site and SDMX api support text/json
        "Accept": "text/json",
        "x-api-key": abs_api_key,
    }
    url = f"https://indicator.data.abs.gov.au/data/{dataflow_id}"
    resp = sdmx.api.read_url(url, headers=headers)
    # surely there has to be a better way to make a nice dataframe than this... TODO FIXME...
    structure = json.loads(resp.response.content)["structure"]["dimensions"]["series"]
    replacements = []
    new_cols = {}
    for d in structure:
        col_name = d.get("id", None)
        replacement_col_name = d.get("name", None)
        for v in d["values"]:
            orig_value = v["id"]
            replacement_value = v["name"]
            replacements.append(
                (col_name, replacement_col_name, orig_value, replacement_value)
            )
            new_cols[col_name] = replacement_col_name
    replacements_df = pd.DataFrame.from_records(
        replacements, columns=("col", "col_replacement", "val", "val_replacement")
    )
    # print(replacements_df)
    df = sdmx.to_pandas(resp, rtype="rows", attributes="osg")
    assert isinstance(df, pd.DataFrame)

    def perform_subst(val, **kwargs):
        d = kwargs.get("kwds")
        return d[val] if val in d else val

    def fix_dataframe(
        df: pd.DataFrame, replacements_df: pd.DataFrame, new_cols: dict
    ) -> pd.DataFrame:
        df = df.reset_index(drop=False)
        cols = [
            new_cols[col_name] if col_name in new_cols else col_name
            for col_name in df.columns
        ]

        # firstly change the values for each column to the recommended values for plotting
        for col_name in df.columns:
            reps = replacements_df[replacements_df["col"] == col_name]
            if len(reps) > 0:
                reps_as_dict = {
                    getattr(t, "val"): getattr(t, "val_replacement")
                    for t in reps.itertuples()
                }
                df[col_name] = df[col_name].apply(perform_subst, kwds=reps_as_dict)
        # finally change the column names to ABS preferred names for the column
        df.columns = cols
        # print(df)

        update_datapoints(df, dataflow_id)
        return df

    return fix_dataframe(df, replacements_df, new_cols)


def dataflows(update: bool = False, abs_api_key: str = None) -> Iterable[ABSDataflow]:
    if update or ABSDataflow.objects.count() < 1:
        if abs_api_key is None:
            abs_api_key = settings.ABS_API_KEY
        headers = {
            "Accept": "application/json",
            "x-api-key": abs_api_key,
        }
        resp = requests.get(
            "https://indicator.data.abs.gov.au/dataflows", headers=headers
        )
        data = json.loads(resp.content)
        assert "resources" in data
        # print(data)
        for k, v in data["references"].items():
            # print(v)
            ABSDataflow.objects.update_or_create(
                abs_id=v["id"], defaults={"abs_id": v["id"], "name": v["name"]}
            )

    return ABSDataflow.objects.all()