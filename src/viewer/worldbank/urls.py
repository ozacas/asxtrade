from django.urls import path
from worldbank.views import (
    worldbank_index_view,
    WorldBankSCMView,
    WorldBankSCSMView,
    WorldBankSCMMView,
    ajax_autocomplete_scm_view,
    ajax_autocomplete_view,
)

urlpatterns = [
    path("", worldbank_index_view, name="worldbank-data-view"),
    path("scsm", WorldBankSCSMView.as_view(), name="worldbank-scsm-view"),
    path("scm", WorldBankSCMView.as_view(), name="worldbank-scm-view"),
    path("scmm", WorldBankSCMMView.as_view(), name="worldbank-scmm-view"),
    path(
        "autocomplete/scsm",
        ajax_autocomplete_view,
        name="ajax-worldbank-scsm-autocomplete",
    ),
    path(
        "autocomplete/scm",
        ajax_autocomplete_scm_view,
        name="ajax-worldbank-scm-autocomplete",
    ),
    path(
        "autocomplete/scmm",
        ajax_autocomplete_view,
        name="ajax-worldbank-scmm-autocomplete",
    ),
]