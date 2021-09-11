from django.urls import path
from investingdotcom.views import (
    BondFormView,
    CryptoFormView,
    CommodityFormView,
    ajax_country_bond_autocomplete,
)

urlpatterns = [
    path(
        "ajax/country-bonds-autocomplete",
        ajax_country_bond_autocomplete,
        name="country-bonds-autocomplete",
    ),
    path("bonds/", BondFormView.as_view(), name="bond-form"),
    path("crypto/", CryptoFormView.as_view(), name="crypto-form"),
    path("commodity/", CommodityFormView.as_view(), name="commodity-form"),
]