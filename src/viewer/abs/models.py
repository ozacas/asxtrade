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
    df = sdmx.to_pandas(resp, rtype="rows", attributes="osg")
    assert isinstance(df, pd.DataFrame)
    resp = requests.get(url, headers=headers)
    tree = json.loads(resp.content)
    import pprint

    print(df)
    pprint.pprint(tree["structure"])
    return df


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