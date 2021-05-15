from django.urls import path
from investingdotcom.views import CryptoFormView, CommodityFormView

urlpatterns = [
    path("crypto/", CryptoFormView.as_view(), name="crypto-form"),
    path("commodity/", CommodityFormView.as_view(), name="commodity-form"),
]