from django.urls import path
from abs.views import (
    abs_index_view
)

urlpatterns = [
    path("", abs_index_view, name="abs-homepage"),
]
