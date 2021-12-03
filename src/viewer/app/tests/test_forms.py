"""
Responsible for testing forms.py and all related state (validation functions etc.)
"""
import pytest
from django.core.exceptions import ValidationError
from app.forms import (
    is_not_blank,
    is_valid_sector,
    MoverSearchForm,
    DividendSearchForm,
    CompanySearchForm,
)
from app.models import stocks_by_sector
from app.tests.test_models import comp_deets # needed to ensure stocks_by_sector() doesnt fail an assertion. pylint: disable=unused-import

def test_is_not_blank():
    with pytest.raises(ValidationError):
        is_not_blank("")
    is_not_blank("hello")
    is_not_blank("0")
    is_not_blank("false")


@pytest.mark.django_db # needs access to the database to validate sector name
def test_is_valid_sector(comp_deets): # pylint: disable=unused-argument,redefined-outer-name
    stocks_by_sector.cache_clear() # prevent cache pollution from ruining test
    for item1, item2 in CompanySearchForm.SECTOR_CHOICES:
        # the database is not populated so we cant check return value, check for an exception
        assert len(item1) == len(item2) and len(item1) > 0
        is_valid_sector(item1)


#@pytest.mark.django_db
#def test_sector_search_form(comp_deets): # pylint: disable=unused-argument,redefined-outer-name
#    fm1 = SectorSearchForm(data={"sector": SectorSearchForm.SECTOR_CHOICES[0][1]})
#    assert fm1.is_valid()
#    fm2 = SectorSearchForm(data={})
#    assert not fm2.is_valid()

def test_mover_search_form():
    fm1 = MoverSearchForm(data={})
    assert not fm1.is_valid()
    fm2 = MoverSearchForm(data={'threshold': 50.0, 'timeframe_in_days': 10, 'metric': 'change_in_percent'})
    assert fm2.is_valid()

def test_dividend_search_form():
    # nothing set is ok for this form
    fm1 = DividendSearchForm(data={})
    assert fm1.is_valid()
    fm2 = DividendSearchForm(data={
        'min_yield': 0.0,
        'max_yield': 10.0,
        'min_pe':    0.0,
        'max_pe':    12.0,
        'min_eps_aud': 0.01
    })
    assert fm2.is_valid()

def test_company_search():
    fm1 = CompanySearchForm(data={})
    assert fm1.is_valid()
    fm2 = CompanySearchForm(data={'name': 'sydney', 'activity': 'air'})
    assert fm2.is_valid()
