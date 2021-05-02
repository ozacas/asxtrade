"""
Responsible for providing django template helpers to get the page rendered
"""
import re
from datetime import datetime
from functools import lru_cache
from django.template.defaulttags import register
from django.templatetags.static import static
from django.utils.safestring import mark_safe
from app.models import CompanyDetails, is_in_watchlist
import numpy as np
import pandas as pd


@register.filter
def get_item(d, key):
    assert isinstance(d, dict)
    assert isinstance(key, str)
    assert key is not None and len(key) > 0
    assert re.match(r"^\w+$", key)
    return d.get(key)


@register.filter
def has_item(d, key):
    try:
        val = get_item(d, key)
        assert val is not None
        return True
    except KeyError:
        return False


@register.filter
def percentage(value):
    try:
        return value * 100.0
    except TypeError:
        return 0.0


@register.filter
@lru_cache(maxsize=1024)
def stock_sector(stock_code):
    assert stock_code is not None and len(stock_code) >= 3
    rec = CompanyDetails.objects.filter(asx_code=stock_code).first()
    if rec is None:
        return ""
    s = rec.sector_name
    return s


@register.simple_tag
def clickable_stock(asx_code: str, **kwargs):
    assert asx_code is not None
    user = kwargs.get("user")
    path = kwargs.get("next")
    assert isinstance(user, str)
    assert path is not None and len(path) > 0
    found = is_in_watchlist(user, asx_code)
    # print("Checking {} is in watchlist for {}: {}".format(asx_code, user, found))
    star_elem = '<a href="/watchlist/{}?next={}">'.format(asx_code, path)
    star_png = '<img src="{}" width="12" />'.format(static("star.png"))
    empty_star_png = '<img src="{}" width="12" />'.format(static("empty-star.png"))
    star_content = star_png if found else empty_star_png
    end_star_elem = "</a>"
    purchase_elem = '<a href="/purchase/{}?next={}">'.format(asx_code, path)
    money_static = static("money.png")
    money_png = '<img src="{}" width="12" />'.format(money_static)
    end_purchase_elem = "</a>&nbsp;"
    content = '<a href="/show/{}">{}</a>'.format(asx_code, asx_code)
    elements = (
        star_elem,
        star_content,
        end_star_elem,
        purchase_elem,
        money_png,
        end_purchase_elem,
        content,
    )
    return mark_safe("".join(elements))


@register.filter
def tabulate_as_html(value):
    assert isinstance(value, pd.DataFrame)
    column_headings = [
        v.strftime("%Y-%m-%d") if isinstance(v, datetime) else v for v in value.columns
    ]
    table_str = "<table>"
    # headings
    table_str += "<tr><th></th>"
    for heading in column_headings:
        table_str += f"<th>{heading}</th>"
    table_str += "</tr>"
    # data rows
    for index, series in value.iterrows():
        table_str += "<tr>"
        table_str += f"<td>{index}</td>"
        for v in series:
            if np.isnan(v):
                table_str += "<td></td>"
            else:
                table_str += f'<td align="right">{v}</td>'
        table_str += "</tr>"
    table_str += "</table>"
    return mark_safe(table_str)
