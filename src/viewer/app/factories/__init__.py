"""
Responsible for providing factories to enable creation of model objects to test the app.
"""
import factory
from pytest_factoryboy import register
from django.conf import settings
from django.contrib.auth import get_user_model
import faker

faker = faker.Faker()


class CompanyDetailsFactory(factory.django.DjangoModelFactory):
    """
       Provide a factory which creates CompanyDetails instances
    """
    class Meta:
        model = "app.CompanyDetails"

    sector_name = factory.Sequence(lambda n: "Financials")
    asx_code = factory.Sequence(lambda n: "ANZ")


class SecurityFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = "app.Security"

    asx_code = "ANZ"
    asx_isin_code = "ISIN000001"


class QuotationFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = "app.Quotation"

    asx_code = factory.Sequence(lambda n: "ANZ")
    error_code = ""  # no error
    fetch_date = faker.date()


class WatchlistFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = "app.Watchlist"

class PurchaseFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = "app.VirtualPurchase"
    