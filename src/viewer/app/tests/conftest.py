"""
Responsile for registering both instance and factory
for unit tests
"""
from pytest_factoryboy import register

from app.factories import (
    CompanyDetailsFactory,
    QuotationFactory,
    SecurityFactory,
    PurchaseFactory,
    WatchlistFactory
)

register(CompanyDetailsFactory)
register(QuotationFactory)
register(SecurityFactory)
register(WatchlistFactory)
register(PurchaseFactory)
