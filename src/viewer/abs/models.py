import django.db.models as model
from djongo.models import ObjectIdField, DjongoManager

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
    scope = model.TextField() # always 'abs' for now

    objects = DjongoManager()

    class Meta:
        db_table = "abs_inverted_index"