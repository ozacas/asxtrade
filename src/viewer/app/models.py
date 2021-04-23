"""
Data models structured to be identical to the format specified by asxtrade.py
and the free ASX API endpoint
"""
from datetime import datetime, timedelta, date
import re
import io
import json
import pytz
import hashlib
from collections import defaultdict
import pandas as pd
from functools import wraps
from time import time
from bson.binary import Binary
import django.db.models as model
from django.conf import settings
from django.forms.models import model_to_dict
from django.contrib.auth import get_user_model
from django.db.models.signals import post_save
from django.core.validators import MaxValueValidator, MinValueValidator
from django.dispatch import receiver
from djongo.models import ObjectIdField, DjongoManager, ArrayField
from djongo.models.json import JSONField
from django.http import Http404
from cachetools import cached, LRUCache, LFUCache, keys, func


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print('func:%r args:[%r, %r] took: %2.4f sec' %  (f.__name__, args, kw, te-ts))
        print("func:{} took: {} sec".format(f.__name__, te - ts))
        return result

    return wrap


watchlist_cache = LRUCache(maxsize=1024)


def validate_stock(stock: str) -> None:
    assert stock is not None
    assert isinstance(stock, str) and len(stock) >= 3
    assert re.match(r"^\w+$", stock)


def validate_date(d: str) -> None:
    assert isinstance(d, str) and len(d) < 20  # YYYY-mm-dd must be less than 20
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", d)


def validate_sector(sector: str) -> None:
    assert len(sector) > 0
    assert sector in set([sector1 for sector1, sector2 in all_sectors()])


def validate_user(user):
    assert user is not None
    assert user.is_active
    assert user.is_authenticated
    assert not user.is_anonymous
    return user  # fluent style convenience


class Timeframe:
    """
    Provide a single object which encapsulates timeframes for the app to perform an an analysis: either current or retrospective.
    In this way, callers can just ask the timeframe for the dates to use, rather than needing to have date code built-in. Four options are available:
    1) no arguments ie. Timeframe() which is past 30 days from today
    2) Timeframe(past_n_days=XXX) for past XXX days from today
    3) Timeframe(from_date='YYYY-mm-dd', to_date='YYYY-mm-dd') - a date range inclusive from from_date to to_date
    4) Timeframe(from_date='YYYY-mm-dd', n=30) eg. 30 days from specified date (inclusive)
    In any case Timeframe.desired_dates() yields a list of date in ascending order
    """

    today = None  # useful for unit testing: must be datetime.date if specified

    def __init__(self, **kwargs):
        d = dict(**kwargs)
        for k in [
            "from_date",
            "to_date",
        ]:  # validate all dates to detect problems early
            if k in d:
                validate_date(d.get(k))
        self.today = d.pop("today", None)  # remove today from being placed into self.tf
        self.tf = d

    def __lt__(self, other):
        a_from = self.tf.get("from_date", None)
        b_from = other.tf.get("from_date", None)
        if a_from is not None and b_from is not None:
            return datetime.strptime(a_from, "%Y-%m-%d") < datetime.strptime(
                b_from, "%Y-%m-%d"
            )
        return 0

    def __hash__(self):
        return hash(tuple(self.tf)) ^ hash(self.tf.values())

    def _is_empty_state(self):
        return self.tf == {}

    def all_dates(self):
        # no arguments for timeframe? assume past 30 days as an application-wide default
        if self._is_empty_state():
            return desired_dates(today=self.today, start_date=30)

        # most common use case: past N days only
        past_n_days = self.tf.get("past_n_days", None)
        if past_n_days:
            return desired_dates(today=self.today, start_date=past_n_days)

        # timeframe of interest: from .. to date
        from_date = self.tf.get("from_date", None)
        to_date = self.tf.get("to_date", None)
        if all([from_date is not None, to_date is not None]):
            print(from_date)
            print(to_date)
            validate_date(from_date)
            validate_date(to_date)
            possible_dates = desired_dates(
                today=datetime.strptime(to_date, "%Y-%m-%d").date(),
                start_date=from_date,
            )
            ret = []
            for d in possible_dates:
                ret.append(d)
                if d == to_date:
                    break
            return ret

        # N days from a start date
        n = self.tf.get("n", None)
        if all([from_date is not None, n is not None]):
            validate_date(from_date)
            assert n > 0
            return desired_dates(today=self.today, start_date=from_date)[:n]

        # otherwise unknown input
        assert False

    def __contains__(self, date_to_find):
        return date_to_find in self.all_dates()

    def __len__(self):
        return self.n_days

    @property
    def n_days(self):
        if "past_n_days" in self.tf:
            return self.tf.get("past_n_days")
        elif "n" in self.tf:
            return self.tf.get("n")
        elif self.tf == {}:
            return 30
        else:
            return len(self.all_dates())  # expensive

    def date_range(self):
        return "{} - {}".format(self.earliest_date, self.most_recent_date)

    @property
    def description(self):
        if "past_n_days" in self.tf or self._is_empty_state():
            return "past {} days since {}".format(
                self.tf.get("past_n_days", self.n_days), self.earliest_date
            )
        elif all(["from_date" in self.tf, "to_date" in self.tf]):
            return "dates {} thru {} (inclusive)".format(
                self.tf.get("from_date"), self.tf.get("to_date")
            )
        else:
            all_dates = self.all_dates()
            return "dates {} thru {} (inclusive)".format(all_dates[0], all_dates[-1])

    @property
    def earliest_date(self):
        from_date = self.tf.get("from_date", None)
        return from_date if from_date is not None else self.all_dates()[0]

    @property
    def most_recent_date(self):
        to_date = self.tf.get("to_date", None)
        return to_date if to_date is not None else self.all_dates()[-1]

    def __str__(self):
        return f"Timeframe: {self.tf}"


class Quotation(model.Model):
    _id = ObjectIdField()
    error_code = model.TextField(max_length=100)  # ignore record iff set to non-empty
    error_descr = model.TextField(max_length=100)
    fetch_date = model.TextField(
        null=False, blank=False
    )  # used as an index (string) always YYYY-mm-dd
    asx_code = model.TextField(blank=False, null=False, max_length=20)
    annual_dividend_yield = model.FloatField()
    average_daily_volume = model.IntegerField()
    bid_price = model.FloatField()
    change_in_percent = model.FloatField()
    change_price = model.FloatField()
    code = model.TextField(blank=False, null=False, max_length=20)
    day_high_price = model.FloatField()
    day_low_price = model.FloatField()
    deprecated_market_cap = (
        model.IntegerField()
    )  # NB: deprecated use market_cap instead
    deprecated_number_of_shares = model.IntegerField()
    descr_full = model.TextField(max_length=100)  # eg. Ordinary Fully Paid
    eps = model.FloatField()
    isin_code = model.TextField(max_length=100)
    last_price = model.FloatField()
    last_trade_date = model.DateField()
    market_cap = model.IntegerField()
    number_of_shares = model.IntegerField()
    offer_price = model.FloatField()
    open_price = model.FloatField()
    pe = model.FloatField()
    previous_close_price = model.FloatField()
    previous_day_percentage_change = model.FloatField()
    suspended = model.BooleanField()
    volume = model.IntegerField()
    year_high_date = model.DateField()
    year_high_price = model.FloatField()
    year_low_date = model.DateField()
    year_low_price = model.FloatField()

    objects = DjongoManager()  # convenient access to mongo API

    def __str__(self):
        assert self is not None
        return str(model_to_dict(self))

    def is_error(self):
        if self.error_code is None:
            return False
        return len(self.error_code) > 0

    def eps_as_cents(self):
        if any([self.is_error(), self.eps is None]):
            return 0.0
        return self.eps * 100.0

    def volume_as_millions(self):
        """
        Return the volume as a formatted string (rounded to 2 decimal place)
        represent the millions of dollars transacted for a given quote
        """
        if any([self.is_error(), self.volume is None, self.last_price is None]):
            return ""

        return "{:.2f}".format(self.volume * self.last_price / 1000000.0)

    class Meta:
        db_table = "asx_prices"
        managed = False  # managed by asxtrade.py


class Security(model.Model):
    # eg. { "_id" : ObjectId("5efe83dd4b1fe020d5ba2de8"), "asx_code" : "T3DAC",
    #  "asx_isin_code" : "AU0000T3DAC0", "company_name" : "333D LIMITED",
    # "last_updated" : ISODate("2020-07-26T00:49:11.052Z"),
    # "security_name" : "OPTION EXPIRING 18-AUG-2018 RESTRICTED" }
    _id = ObjectIdField()
    asx_code = model.TextField(blank=False, null=False)
    asx_isin_code = model.TextField()
    company_name = model.TextField()
    last_updated = model.DateField()
    security_name = model.TextField()

    objects = DjongoManager()

    class Meta:
        db_table = "asx_isin"
        managed = False  # managed by asxtrade.py
        verbose_name = "Security"
        verbose_name_plural = "Securities"


class CompanyDetails(model.Model):
    # { "_id" : ObjectId("5eff01d14b1fe020d5453e8f"), "asx_code" : "NIC", "delisting_date" : null,
    # "fax_number" : "02 9221 6333", "fiscal_year_end" : "31/12", "foreign_exempt" : false,
    # "industry_group_name" : "Materials",
    # "latest_annual_reports" : [ { "id" : "02229616", "document_release_date" : "2020-04-29T14:45:12+1000",
    # "document_date" : "2020-04-29T14:39:36+1000", "url" : "http://www.asx.com.au/asxpdf/20200429/pdf/44hc5731pmh9mw.pdf", "relative_url" : "/asxpdf/20200429/pdf/44hc5731pmh9mw.pdf", "header" : "Annual Report and Notice of AGM", "market_sensitive" : false, "number_of_pages" : 118, "size" : "4.0MB", "legacy_announcement" : false }, { "id" : "02209126", "document_release_date" : "2020-02-28T18:09:26+1100", "document_date" : "2020-02-28T18:06:25+1100", "url" : "http://www.asx.com.au/asxpdf/20200228/pdf/44fm8tp5qy0k7x.pdf", "relative_url" : "/asxpdf/20200228/pdf/44fm8tp5qy0k7x.pdf",
    # "header" : "Annual Report and Appendix 4E", "market_sensitive" : true, "number_of_pages" : 64,
    # "size" : "1.6MB", "legacy_announcement" : false }, { "id" : "02163933", "document_release_date" :
    # "2019-10-25T11:50:50+1100", "document_date" : "2019-10-25T11:48:43+1100",
    # "url" : "http://www.asx.com.au/asxpdf/20191025/pdf/449w6d0phvgr05.pdf",
    # "relative_url" : "/asxpdf/20191025/pdf/449w6d0phvgr05.pdf", "header" : "Annual Report and Notice of AGM",
    # "market_sensitive" : false, "number_of_pages" : 74, "size" : "2.5MB", "legacy_announcement" : false } ],
    # "listing_date" : "2018-08-20T00:00:00+1000",
    # "mailing_address" : "Level 2, 66 Hunter Street, SYDNEY, NSW, AUSTRALIA, 2000",
    # "name_abbrev" : "NICKELMINESLIMITED", "name_full" : "NICKEL MINES LIMITED",
    # "name_short" : "NICKLMINES", "phone_number" : "02 9300 3311",
    # "primary_share" : { "code" : "NIC", "isin_code" : "AU0000018236", "desc_full" : "Ordinary Fully Paid",
    # "last_price" : 0.61, "open_price" : 0.595, "day_high_price" : 0.615, "day_low_price" : 0.585,
    # "change_price" : 0.02, "change_in_percent" : "3.39%", "volume" : 3127893, "bid_price" : 0.605,
    # "offer_price" : 0.61, "previous_close_price" : 0.59, "previous_day_percentage_change" : "1.724%",
    # "year_high_price" : 0.731, "last_trade_date" : "2020-07-24T00:00:00+1000",
    # "year_high_date" : "2019-09-24T00:00:00+1000", "year_low_price" : 0.293,
    # "year_low_date" : "2020-03-18T00:00:00+1100", "pe" : 8.73, "eps" : 0.0699,
    # "average_daily_volume" : 7504062, "annual_dividend_yield" : 0, "market_cap" : 1255578789,
    # "number_of_shares" : 2128099642, "deprecated_market_cap" : 1127131000,
    # "deprecated_number_of_shares" : 1847755410, "suspended" : false,
    # "indices" : [ { "index_code" : "XKO", "name_full" : "S&P/ASX 300", "name_short" : "S&P/ASX300",
    # "name_abrev" : "S&P/ASX 300" }, { "index_code" : "XAO", "name_full" : "ALL ORDINARIES",
    # "name_short" : "ALL ORDS", "name_abrev" : "All Ordinaries" }, { "index_code" : "XSO",
    # "name_full" : "S&P/ASX SMALL ORDINARIES", "name_short" : "Small Ords",
    # "name_abrev" : "S&P/ASX Small Ords" }, { "index_code" : "XMM",
    # "name_full" : "S&P/ASX 300 Metals and Mining (Industry)", "name_short" : "MTL&MINING",
    # "name_abrev" : "Metals and Mining" } ] }, "primary_share_code" : "NIC",
    # "principal_activities" : "Nickel pig iron and nickel ore production.",
    # "products" : [ "shares" ], "recent_announcement" : false,
    # "registry_address" : "Level 3, 60 Carrington Street, SYDNEY, NSW, AUSTRALIA, 2000",
    # "registry_name" : "COMPUTERSHARE INVESTOR SERVICES PTY LIMITED",
    # "registry_phone_number" : "02 8234 5000", "sector_name" : "Metals & Mining",
    # "web_address" : "http://www.nickelmines.com.au/" }
    _id = ObjectIdField()
    asx_code = model.TextField(null=False, blank=False)
    delisting_date = model.TextField(null=True)  # if null, not delisted
    name_full = model.TextField(null=False, blank=False)
    phone_number = model.TextField(null=False, blank=True)
    bid_price = model.FloatField()
    offer_price = model.FloatField()
    latest_annual_reports = JSONField()
    previous_close_price = model.FloatField()
    average_daily_volume = model.IntegerField()
    number_of_shares = model.IntegerField()
    suspended = model.BooleanField()
    indices = JSONField()
    primary_share_code = model.TextField()
    principal_activities = model.TextField()
    products = JSONField()
    recent_announcement = model.BooleanField()
    registry_address = model.TextField()
    registry_name = model.TextField()
    registry_phone_number = model.TextField()
    sector_name = model.TextField()
    web_address = model.TextField()

    objects = DjongoManager()

    class Meta:
        managed = False
        db_table = "asx_company_details"
        verbose_name = "Company Detail"
        verbose_name_plural = "Company Details"


class Watchlist(model.Model):
    id = ObjectIdField(unique=True, db_column="_id")
    # record stocks of interest to the user
    user = model.ForeignKey(settings.AUTH_USER_MODEL, on_delete=model.CASCADE)
    asx_code = model.TextField()
    when = model.DateTimeField(auto_now_add=True)

    objects = DjongoManager()

    class Meta:
        managed = True  # viewer application is responsible NOT asxtrade.py
        db_table = "user_watchlist"


@func.lru_cache(maxsize=16)
def find_user(username: str):
    return get_user_model().objects.filter(username=username).first()


def hash_watchlist_key(username, asx_code, **kwargs):  # pylint: disable=unused-argument
    return keys.hashkey(username, asx_code)


@cached(watchlist_cache, key=hash_watchlist_key)
def is_in_watchlist(username: str, asx_code: str) -> bool:
    user = find_user(username)
    if user is None:
        return False
    else:
        rec = Watchlist.objects.filter(asx_code=asx_code, user=user).first()
        return rec is not None


def toggle_watchlist_entry(user, asx_stock):
    assert user is not None
    assert isinstance(asx_stock, str)
    current_watchlist = user_watchlist(user)
    # 1. update db
    if asx_stock in current_watchlist:  # remove from watchlist?
        Watchlist.objects.filter(user=user, asx_code=asx_stock).delete()
    else:
        Watchlist(user=user, asx_code=asx_stock).save()
    # 2. invalidate from watchlist_cache as wrong cached value may be stored already
    try:
        # NB: must be the same key calculation and args as is_in_watchlist()
        watchlist_cache.pop(key=hash_watchlist_key(user.username, asx_stock))
    except KeyError:  # not in cache? thats ok so...
        pass


def user_watchlist(user):
    """
    Given a user object eg. from find_user() return the set of stock codes which the user
    has indicated an interest in
    """
    hits = Watchlist.objects.filter(user=user).values_list("asx_code", flat=True)
    results = set(hits)
    # if logger:
    #    logger.info("Found {} stocks in watchlist for user=".format(len(results), user.username))
    return results


@func.lru_cache(maxsize=16)
def all_available_dates(reference_stock="ANZ"):
    """
    Returns a sorted list of available dates where the reference stock has a price. stocks
    which are suspended/delisted may have limited dates. The list is sorted from oldest to 
    newest (ascending sort). As this is a frequently used query, an LRU cache is implemented 
    to avoid hitting the database too much.
    """
    # use reference_stock to quickly search the db by limiting the stocks searched
    dates = Quotation.objects.mongo_distinct(
        "fetch_date", {"asx_code": reference_stock}
    )
    ret = sorted(dates, key=lambda k: datetime.strptime(k, "%Y-%m-%d"))
    return ret


def stock_info(stock: str, warning_cb=None) -> dict:
    assert len(stock) > 0
    securities = Security.objects.filter(asx_code=stock)
    if securities is None and warning_cb:
        warning_cb(f"No securities available for {stock}")
    company_details = CompanyDetails.objects.filter(asx_code=stock).first()
    if company_details is None:
        if warning_cb:
            warning_cb(f"No details available for {stock}")
        result = {}
    else:
        result = model_to_dict(company_details)
    result["securities"] = securities
    return result


@func.lru_cache(maxsize=1)
def stocks_by_sector() -> pd.DataFrame:
    rows = [
        d
        for d in CompanyDetails.objects.values("asx_code", "sector_name").order_by(
            "asx_code"
        )
    ]
    df = pd.DataFrame.from_records(rows)
    assert len(df) > 0
    colnames = set(df.columns)
    assert "asx_code" in colnames and "sector_name" in colnames
    return df


class Sector(model.Model):
    """
    Table of ASX sector (GICS) names. Manually curated for now.
    """

    id = ObjectIdField(unique=True, db_column="_id")
    sector_name = model.TextField(unique=True)
    sector_id = model.IntegerField(db_column="id")

    objects = DjongoManager()

    class Meta:
        managed = False
        db_table = "sector"


@func.lru_cache(maxsize=1)
def all_sectors() -> list:
    iterable = list(
        CompanyDetails.objects.order_by()
        .values_list("sector_name", flat=True)
        .distinct()
    )
    # print(iterable)
    results = [
        (sector, sector) for sector in iterable
    ]  # as tuples since we want to use it in django form choice field
    return results


def companies_with_same_sector(stock: str) -> set:
    """
    Return the set of all known companies designated with the same sector as the specified stock
    """
    cd = CompanyDetails.objects.filter(asx_code=stock).first()
    if cd is None:
        return set()
    return all_sector_stocks(cd.sector_name)


@func.lru_cache(maxsize=16)
def all_sector_stocks(sector_name: str) -> set:
    """
    Return a set of ASX stock codes for every security designated as part of the specified sector
    """
    assert sector_name is not None and len(sector_name) > 0
    ss = stocks_by_sector()
    ss = ss[ss["sector_name"] == sector_name]
    stocks = set(ss["asx_code"])
    return stocks


@func.lfu_cache(maxsize=2)  # cache today's data only to save memory
def valid_quotes_only(ymd: str, sort_by=None, ensure_date_has_data=True) -> tuple:
    validate_date(ymd)
    results = (
        Quotation.objects.filter(fetch_date=ymd)
        .exclude(asx_code__isnull=True)
        .exclude(error_code="id-or-code-invalid")
        .exclude(last_price__isnull=True)
        .order_by(
            "-annual_dividend_yield", "-last_price", "-volume"
        )  # default order_by, see below
    )
    if len(results) == 0 and ensure_date_has_data:
        # decrease date by 1 and try again... (we cant increment because this might go into the future)
        dt = datetime.strptime(ymd, "%Y-%m-%d") - timedelta(days=1)
        return valid_quotes_only(
            dt.strftime("%Y-%m-%d"), sort_by=sort_by, ensure_date_has_data=True
        )

    if sort_by is not None:
        results = results.order_by(sort_by)
    assert results is not None  # POST-CONDITION: must be valid queryset
    return results, ymd


def desired_dates(
    today=None, start_date=None
):  # today is provided as keyword arg for testing
    """
    Return a list of contiguous dates from [today-n_days thru to today inclusive] as 'YYYY-mm-dd' strings, Ordered
    from start_date thru today inclusive. Start_date may be:
    1) a string in YYYY-mm-dd format OR
    2) a datetime instance OR
    3) a integer number of days (>0) from today backwards to return.
    """
    if today is None:
        today = date.today()
    if isinstance(start_date, (datetime, date)):
        pass  # FALLTHRU
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    elif isinstance(start_date, int):
        assert start_date > 0
        start_date = today - timedelta(days=start_date - 1)  # -1 for today inclusive
    else:
        raise ValueError("Unsupported start_date {}".format(type(start_date)))
    assert start_date <= today
    all_dates = [
        d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, today, freq="D")
    ]
    assert len(all_dates) > 0
    return sorted(all_dates, key=lambda d: datetime.strptime(d, "%Y-%m-%d"))


@func.lru_cache(maxsize=2)
def all_stocks(strict=True):
    """Return all securities known (even if not stocks) if strict=False, otherwise ordinary fully paid shares and ETFs only"""
    if strict:
        all_securities = []
        for security in Security.objects.all():
            name = security.security_name.lower()
            if "etf" in name or "ordinary" in name:
                # print(name)
                all_securities.append(security.asx_code)
    else:
        all_securities = Security.objects.values_list("asx_code", flat=True)

    return set(all_securities)


def find_movers(
    threshold,
    timeframe: Timeframe,
    increasing=True,
    decreasing=False,
    max_price=None,
    field="change_in_percent",
):
    """
    Return a dataframe with row index set to ASX ticker symbols and the only column set to 
    the sum over all desired dates for percentage change in the stock price (by default). A negative sum
    implies a decrease, positive an increase in price over the observation period. Some fields are percentages, some AUD cents etc (it is up to the user to get it right)
    """
    assert threshold >= 0.0
    cip = company_prices(all_stocks(), timeframe, fields=field, missing_cb=None)

    if field == "change_in_percent":
        movements = cip.sum(axis=0)
    else:
        movements = cip.diff(periods=1, axis=0).fillna(0.0).sum(axis=0)
    # FALLTHRU...
    results = movements[movements.abs() >= threshold]
    print(
        "Found {} movers (using {}) before filtering: {} {}".format(
            len(results), field, increasing, decreasing
        )
    )
    if not increasing:
        results = results.drop(results[results > 0.0].index)
    if not decreasing:
        results = results.drop(results[results < 0.0].index)
    # print(results)
    if max_price is not None:
        ymd = latest_quotation_date("ANZ")
        qs, actual_ymd = valid_quotes_only(ymd, ensure_date_has_data=True)
        stocks_lte_max_price = [q.asx_code for q in qs if q.last_price <= max_price]
        results = results.filter(stocks_lte_max_price)
    print("Reporting {} movers after filtering".format(len(results)))
    return results


def find_named_companies(wanted_name, wanted_activity):
    ret = set()
    if len(wanted_name) > 0:
        # match by company name first...
        ret.update(
            CompanyDetails.objects.filter(name_full__icontains=wanted_name).values_list(
                "asx_code", flat=True
            )
        )
        # but also matching codes
        ret.update(
            CompanyDetails.objects.filter(asx_code__icontains=wanted_name).values_list(
                "asx_code", flat=True
            )
        )
        # and if the code has no details, we try another method to find them...
        # by checking all codes seen on the most recent date
        latest_date = latest_quotation_date("ANZ")
        ret.update(
            Quotation.objects.filter(asx_code__icontains=wanted_name)
            .filter(fetch_date=latest_date)
            .values_list("asx_code", flat=True)
        )

    if len(wanted_activity) > 0:
        ret.update(
            CompanyDetails.objects.filter(
                principal_activities__icontains=wanted_activity
            ).values_list("asx_code", flat=True)
        )
    return ret


def latest_quotation_date(stock):
    d = all_available_dates(reference_stock=stock)
    return d[-1]


def latest_quote(stocks):
    """
    If stocks is a str, retrieves the latest quote and returns a tuple (Quotation, latest_date).
    If stocks is None, returns a tuple (queryset, latest_date) of all stocks.
    If stocks is an iterable, returns a tuple (queryset, latest_date) of selected stocks
    """
    if isinstance(stocks, str):
        latest_date = latest_quotation_date(stocks)
        obj = Quotation.objects.get(asx_code=stocks, fetch_date=latest_date)
        return (obj, latest_date)
    else:
        latest_date = latest_quotation_date("ANZ")
        qs = Quotation.objects.filter(fetch_date=latest_date)
        if stocks is not None:
            if len(stocks) == 1:
                qs = qs.filter(asx_code=stocks[0])
            else:
                qs = qs.filter(asx_code__in=stocks)
        return (qs, latest_date)


def selected_cached_stocks_cip(stocks, timeframe: Timeframe) -> pd.DataFrame:
    n = len(stocks)
    assert n > 0
    all_cip = cached_all_stocks_cip(timeframe)
    result_df = all_cip.filter(items=stocks, axis=0)
    got = len(result_df)
    print("Selected stocks cip: found {} stocks (ie. rows)".format(got))
    assert got <= n
    return result_df


@func.lfu_cache(maxsize=4)
def cached_all_stocks_cip(timeframe: Timeframe):
    all_stocks_cip = company_prices(
        None, timeframe, fields="change_in_percent", missing_cb=None, transpose=True
    )
    return all_stocks_cip


# NB: careful sizing the cache - dont want to use too much memory!
dataframe_in_memory_cache = LFUCache(maxsize=100)


def get_dataframe(tag: str, stocks, debug=False) -> pd.DataFrame:
    """
    To save reading parquet files and constructing each pandas dataframe, we cache all that logic
    so that repeated requests for a given tag dont hit the database. Hopefully.
    """

    def finalise_dataframe(df):
        """Ensure every dataframe, whether cached or not, is the same"""
        if len(df) == 0:  # dont return empty dataframes
            return None
        # print(tag, " ", stocks, " ", df.columns)
        if tag.startswith("uber") and stocks is not None:
            is_desired_stock = df["asx_code"].isin(set(stocks))
            return df[is_desired_stock]
        elif df.columns.name == "asx_code" and stocks is not None:
            return df.filter(items=stocks, axis="columns")
        else:
            return df

    # 1. already cached the dataframe?
    df = dataframe_in_memory_cache.get(tag, None)
    if df is not None:
        return finalise_dataframe(df)

    if debug:
        print(f"{tag} not in memory cache")
    parquet_blob = get_parquet(tag)
    if parquet_blob is None:
        return None
    if debug:
        print(f"{tag} loaded from DB/parquet cache")
        assert isinstance(parquet_blob, bytes)
        print(f"Parsing parquet for {tag}")

    with io.BytesIO(parquet_blob) as fp:
        df = pd.read_parquet(fp)
        dataframe_in_memory_cache[tag] = df
        return finalise_dataframe(df)


def make_superdf(required_tags, stock_codes):
    assert required_tags is not None and len(required_tags) >= 1
    assert stock_codes is None or len(stock_codes) > 0  # NB: zero stocks considered bad
    dataframes = filter(
        lambda df: df is not None,
        [get_dataframe(tag, stock_codes) for tag in required_tags],
    )
    superdf = pd.concat(dataframes, axis=0)
    return superdf


def day_low_high(stock, all_dates=None):
    """
    For the specified dates (specified in strict YYYY-mm-dd format) return
    the day low/high price, last price and volume as a pandas dataframe with dates down and
    columns which represent the prices and volume (in $AUD millions).
    """
    assert isinstance(stock, str) and len(stock) >= 3

    quotes = (
        Quotation.objects.filter(asx_code=stock)
        .filter(fetch_date__in=all_dates)
        .exclude(error_code="id-or-code-invalid")
    )
    rows = []
    for q in quotes:
        rows.append(
            {
                "date": q.fetch_date,
                "day_low_price": q.day_low_price,
                "day_high_price": q.day_high_price,
                "volume": q.volume_as_millions(),
                "last_price": q.last_price,
            }
        )
    day_low_high_df = pd.DataFrame.from_records(rows)
    day_low_high_df.set_index(day_low_high_df["date"], inplace=True)
    return day_low_high_df


def impute_missing(df, method="linear"):
    assert df is not None
    # print("impute_missing: ", df)
    if method == "linear":  # faster...
        result = df.interpolate(
            method=method, limit_direction="forward", axis="columns"
        )
        return result
    else:
        # must have a DateTimeIndex so...
        df.columns = pd.to_datetime(df.columns)
        df = df.interpolate(method=method, limit_direction="forward", axis="columns")
        df.set_index(
            df.index.format(), inplace=True
        )  # convert back to strings for caller compatibility
        return df


@func.lru_cache(maxsize=1)
def all_etfs():
    etf_codes = [
        s.asx_code
        for s in Security.objects.filter(
            security_name="EXCHANGE TRADED FUND UNITS FULLY PAID"
        )
    ]
    print("Found {} ETF codes".format(len(etf_codes)))
    return etf_codes


def increasing_eps(stock_codes, past_n_days=300):
    tf = Timeframe(past_n_days=past_n_days)
    return increasing_only_filter(stock_codes, tf, "eps")


def increasing_yield(stock_codes, past_n_days=300):
    tf = Timeframe(past_n_days=past_n_days)
    return increasing_only_filter(
        stock_codes, tf, "annual_dividend_yield", min_value=0.01
    )


def increasing_only_filter(
    stock_codes, timeframe: Timeframe, field: str, min_value=0.02
):
    assert min_value >= 0.0
    assert timeframe is not None

    if timeframe.n_days < 14:
        raise Http404(
            "Not enough days requested to produce meaningful results: {}".format(
                timeframe.n_days
            )
        )

    # NB: we dont care here if some tags cant be found
    df = company_prices(stock_codes, timeframe, field, transpose=True).fillna(0.0)
    # df will be very large: 300 days * ~2000 stocks... but mostly the numbers will be the same each day...
    # at least 2c per share positive max(eps) is required to be considered significant
    ret = []
    for idx, series in df.iterrows():
        # print(series)
        if series.is_monotonic_increasing and max(series) >= min_value:
            ret.append(idx)

    return ret


def get_required_tags(all_dates, fields):
    required_tags = set()
    for d in all_dates:
        validate_date(d)
        yyyy = d[0:4]
        mm = d[5:7]
        required_tags.add("{}-{}-{}-asx".format(fields, mm, yyyy))
    return required_tags


def first_arg_only(*args):
    return keys.hashkey(args[0])


def rsi_data(stock: str, timeframe: Timeframe):
    stock_df = company_prices(
        [stock],
        timeframe,
        fields=["last_price", "volume", "day_low_price", "day_high_price"],
        missing_cb=None,
    )
    n_dates = len(stock_df)
    if (
        n_dates < 14
    ):  # RSI requires at least 14 prices to plot so reject recently added stocks
        raise Http404(
            "Insufficient price quotes for {} - only {}".format(stock, n_dates)
        )
    # print(stock_df)
    return stock_df


@timing
def company_prices(
    stock_codes,
    timeframe: Timeframe,
    fields="last_price",
    missing_cb=impute_missing,  # or None if you want missing values
    transpose=False,  # return with stocks as columns (default) or rows?
):
    """
    Return a dataframe with the required companies (iff quoted) over the
    specified dates. By default last_price is provided. Fields may be a list,
    in which case the dataframe has columns for each field and dates are rows (in this case only one stock is permitted)
    """
    print(
        "company_prices(len(stocks) == {}, {}, {}, {}, {})".format(
            len(stock_codes) if stock_codes is not None else stock_codes,
            timeframe.description,
            fields,
            missing_cb,
            transpose,
        )
    )

    def prepare_dataframe(df, iterable_of_fields):
        assert isinstance(df, pd.DataFrame)
        is_ok_field = df["field_name"].isin(iterable_of_fields)
        df = df[is_ok_field]
        return df.pivot(index="fetch_date", columns="field_name", values="field_value")

    if not isinstance(fields, str):  # assume iterable if not str...
        assert len(stock_codes) == 1
        tags = get_required_tags(timeframe.all_dates(), "uber")
        result_df = make_superdf(tags, stock_codes)
        result_df = prepare_dataframe(result_df, fields)

        assert set(result_df.columns) == set(fields) or len(result_df) == 0
        # reject dates (ie. rows) which are all NA to avoid downstream problems eg. plotting stocks
        # NB: we ONLY do this for the multi-field case, single field it is callers responsibility
        result_df = result_df.dropna(how="all")
        if transpose:
            return result_df.transpose()
        return result_df

    # print(stock_codes)
    assert isinstance(fields, str)
    all_dates = timeframe.all_dates()
    required_tags = get_required_tags(all_dates, "uber")
    # print(required_tags)
    # construct a "super" dataframe from the constituent parquet data
    superdf = make_superdf(required_tags, stock_codes)
    if (
        "field_name" in superdf.columns
    ):  # optional to permit easier mocking of make_superdf()
        superdf = superdf[superdf["field_name"] == fields]
        superdf = superdf.pivot(
            index="fetch_date", columns="asx_code", values="field_value"
        )

    # drop dates not present in all_dates to ensure we are giving just the results requested
    superdf = superdf.filter(items=all_dates, axis="index")

    # dont transpose for performance by default
    if transpose:
        superdf = superdf.transpose()

    # impute missing if caller wants it (and missing values present)
    if missing_cb is not None and superdf.isnull().values.any():
        if missing_cb == impute_missing and fields == "change_in_percent":
            print(
                "WARNING: fields == change_in_percent with impute_missing() is likely nonsensical"
            )
        superdf = missing_cb(superdf)
    return superdf


class MarketDataCache(model.Model):
    # { "_id" : ObjectId("5f44c54457d4bb6dfe6b998f"), "scope" : "all-downloaded",
    # "tag" : "change_price-05-2020-asx", "dataframe_format" : "parquet",
    # "field" : "change_price", "last_updated" : ISODate("2020-08-25T08:01:08.804Z"),
    # "market" : "asx", "n_days" : 3, "n_stocks" : 0,
    # "sha256" : "75d0ad7e057621e6a73508a178615bcc436d97110bcc484f1cfb7d478475abc5",
    # "size_in_bytes" : 2939, "status" : "INCOMPLETE" }
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
        managed = False  # table is managed by persist_dataframes.py
        db_table = "market_quote_cache"


class VirtualPurchase(model.Model):
    id = ObjectIdField(unique=True, db_column="_id")
    user = model.ForeignKey(settings.AUTH_USER_MODEL, on_delete=model.CASCADE)
    asx_code = model.TextField(max_length=10)
    buy_date = model.DateField()
    price_at_buy_date = model.FloatField()
    amount = model.FloatField()  # dollar value purchased (assumes no fees)
    n = (
        model.IntegerField()
    )  # number of shares purchased at buy_date (rounded down to nearest whole share)

    objects = DjongoManager()

    def current_price(self):
        assert self.n > 0
        quote, _ = latest_quote(self.asx_code)
        if quote is None:
            raise ValueError()
        buy_price = self.price_at_buy_date
        pct_move = (
            (quote.last_price / buy_price) * 100.0 - 100.0 if buy_price > 0.0 else 0.0
        )
        return (self.n * quote.last_price, pct_move)

    def __str__(self):
        cur_price, pct_move = self.current_price()
        return "Purchase on {}: ${} ({} shares@${:.2f}) is now ${:.2f} ({:.2f}%)".format(
            self.buy_date,
            self.amount,
            self.n,
            self.price_at_buy_date,
            cur_price,
            pct_move,
        )

    class Meta:
        managed = True  # viewer application
        db_table = "virtual_purchase"


def user_purchases(user):
    """
    Returns a dict: asx_code -> VirtualPurchase of the specified user's watchlist
    """
    validate_user(user)
    watchlist = user_watchlist(user)
    purchases = defaultdict(list)
    for purchase in VirtualPurchase.objects.filter(user=user):
        # "ghost" purchases are ignored: they are those purchases for stocks that are no longer part of the user's watchlist. And no longer part of purchase performance until placed (again) into the watchlist.
        code = purchase.asx_code
        if not code in watchlist:
            continue
        purchases[code].append(purchase)
    # print("Found virtual purchases for {} stocks".format(len(purchases)))
    return purchases


# class Profile(model.Model):
#     """User profile model for defaults for various analyses and visualisations"""
#     user = model.OneToOneField(get_user_model(), on_delete=model.CASCADE)
#     timeframe_in_days = model.IntegerField(validators=[MinValueValidator(10), MaxValueValidator(100000)], default=180) # for plots/analyses were default is permissable
#     line_size = model.FloatField(validators=[MinValueValidator(0.5), MaxValueValidator(10.0)], default=1.5) # for line based plots

#     objects = DjongoManager()

#     class Meta:
#         db_table = "user_profiles"

# @receiver(post_save, sender=get_user_model())
# def create_user_profile(sender, instance, created, **kwargs):
#     if not created:
#         Profile.objects.create(user=instance)

# @receiver(post_save, sender=get_user_model())
# def save_user_profile(sender, instance, **kwargs):
#     instance.profile.save()


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


@func.lru_cache(maxsize=100)
def get_parquet(tag: str, model=MarketDataCache) -> pd.DataFrame:
    assert len(tag) > 0
    cache_entry = model.objects.filter(tag=tag).first()
    if cache_entry is not None:
        return cache_entry.dataframe
    return None


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
    indicator_id = model.TextField() # xref into WorldBankIndicators.wb_id
    indicator_name = model.TextField() # likewise to name field

    objects = DjongoManager()

    class Meta:
        db_table = "world_bank_inverted_index"
        verbose_name_plural = "World Bank Inverted Indexes"

class WorldBankDataCache(model.Model): # a minimal model because i stuffed up the table name when ingesting worldbank data
    tag = model.TextField(null=False)
    dataframe = model.BinaryField()

    class Meta:
        db_table = "market_data_cache"

class WorldBankIndicators(model.Model):
    id = ObjectIdField(primary_key=True, db_column='_id')
    wb_id = model.TextField()  # world bank id, not to be confused with django id/pk
    name = model.TextField()
    last_updated = model.DateTimeField(auto_now_add=True)
    unit = model.TextField()
    source = JSONField()
    source_note = model.TextField()
    topics = ArrayField(WorldBankTopic)
    source_organisation = model.TextField()
    last_successful_data = model.DateTimeField(null=True)
    last_error_when = model.DateTimeField(null=True) # date of last ingest error for this indicator (or None if not known)
    last_error_msg = model.TextField()  # eg. Error code 175
    last_error_type = model.TextField() # eg. class RuntimeError

    objects = DjongoManager()

    def __init__(self, *args, **kwargs):
        wb_id = kwargs.pop("id", None)
        source_organisation = kwargs.pop("sourceOrganization", None)
        source_note = kwargs.pop("sourceNote", None)
        # wb_id=wb_id, source_organisation=source_organisation, source_note=source_note
        super(WorldBankIndicators, self).__init__(*args, **kwargs)

    def __str__(self):
        return f"{self.wb_id} {self.name} {self.source} last_error_when={self.last_error_when} last_updated={self.last_updated}"

    @property
    def tag(self): # NB: must match the ingest_worldbank_datasets.py tag name...
        return f"{self.wb_id}-yearly-dataframe"

    @property
    def has_data(self):
        obj = MarketDataCache.objects.filter(tag=self.tag).first()
        return obj is not None

    def fetch_data(self) -> pd.DataFrame:
        t = self.tag
        print(f"Fetching parquet dataframe for {t}")
        return get_parquet(t, model=WorldBankDataCache)
   
    class Meta:
        db_table = "world_bank_indicators"
        verbose_name = 'World Bank Metric'
        verbose_name_plural = 'World Bank Metrics'

