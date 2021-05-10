from django.urls import path
from ecb.views import ecb_index_view, ECBDataflowView

urlpatterns = [
    path("", ecb_index_view, name="ecb-index-view"),
    path(
        "dataflow/<str:dataflow>", ECBDataflowView.as_view(), name="ecb-dataflow-view"
    ),
]