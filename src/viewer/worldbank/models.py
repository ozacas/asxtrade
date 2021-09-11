import django.db.models as model
from djongo.models import ObjectIdField, DjongoManager, ArrayField
from djongo.models.json import JSONField
import pandas as pd


def get_parquet(tag: str) -> pd.DataFrame:
    assert len(tag) > 0
    cache_entry = WorldBankDataCache.objects.filter(tag=tag).first()
    if cache_entry is not None:
        return cache_entry.dataframe
    return None


class WorldBankCountry(model.Model):
    """
    Not strictly a country, can include regions eg. africa ex-warzones
    """

    id = ObjectIdField(primary_key=True, db_column="_id")
    country_code = model.TextField()
    name = model.TextField()
    last_updated = model.DateTimeField(auto_now_add=True)

    objects = DjongoManager()

    class Meta:
        db_table = "world_bank_countries"
        verbose_name_plural = "World Bank Countries"


class WorldBankTopic(model.Model):
    id = model.IntegerField(null=False, blank=False, primary_key=True)
    topic = model.TextField(null=False, blank=False)
    source_note = model.TextField(null=False, blank=False)
    last_updated = model.DateTimeField(auto_now_add=True)

    objects = DjongoManager()

    def __init__(self, *args, **kwargs):  # support initialization via ArrayField
        topic = kwargs.pop("value", None)
        source_note = kwargs.pop("sourceNote", None)
        extra_args = {}
        if topic is not None:
            extra_args["topic"] = topic
        if source_note is not None:
            extra_args["source_note"] = source_note
        super(WorldBankTopic, self).__init__(*args, **kwargs, **extra_args)

    class Meta:
        db_table = "world_bank_topics"


class WorldBankInvertedIndex(model.Model):
    id = ObjectIdField(primary_key=True, db_column="_id")
    country = model.TextField()
    topic_id = model.IntegerField()
    topic_name = model.TextField()
    n_attributes = model.IntegerField()
    last_updated = model.DateTimeField()
    tag = model.TextField()
    indicator_id = model.TextField()  # xref into WorldBankIndicators.wb_id
    indicator_name = model.TextField()  # likewise to name field

    objects = DjongoManager()

    class Meta:
        db_table = "world_bank_inverted_index"
        verbose_name_plural = "World Bank Inverted Indexes"


class WorldBankDataCache(model.Model):
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
        db_table = "worldbank_data_cache"


class WorldBankIndicators(model.Model):
    id = ObjectIdField(primary_key=True, db_column="_id")
    wb_id = model.TextField()  # world bank id, not to be confused with django id/pk
    name = model.TextField()
    last_updated = model.DateTimeField(auto_now_add=True)
    unit = model.TextField()
    source = JSONField()
    source_note = model.TextField()
    topics = ArrayField(WorldBankTopic)
    source_organisation = model.TextField()
    last_successful_data = model.DateTimeField(null=True)
    last_error_when = model.DateTimeField(
        null=True
    )  # date of last ingest error for this indicator (or None if not known)
    last_error_msg = model.TextField()  # eg. Error code 175
    last_error_type = model.TextField()  # eg. class RuntimeError

    objects = DjongoManager()

    def __init__(self, *args, **kwargs):
        self.wb_id = kwargs.pop("id", None)
        self.source_organisation = kwargs.pop("sourceOrganization", None)
        self.source_note = kwargs.pop("sourceNote", None)
        # wb_id=wb_id, source_organisation=source_organisation, source_note=source_note
        super(WorldBankIndicators, self).__init__(*args, **kwargs)

    def __str__(self):
        return f"{self.wb_id} {self.name} {self.source} last_error_when={self.last_error_when} last_updated={self.last_updated}"

    @property
    def tag(self):  # NB: must match the ingest_worldbank_datasets.py tag name...
        return f"{self.wb_id}-yearly-dataframe"

    @property
    def has_data(self):
        obj = WorldBankDataCache.objects.filter(tag=self.tag).first()
        return obj is not None

    def fetch_data(self) -> pd.DataFrame:
        t = self.tag
        print(f"Fetching parquet dataframe for {t}")
        return get_parquet(t)

    class Meta:
        db_table = "world_bank_indicators"
        verbose_name = "World Bank Metric"
        verbose_name_plural = "World Bank Metrics"
